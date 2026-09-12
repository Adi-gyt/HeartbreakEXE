"""
Heartbreak.exe - Wallpaper / Lifecycle Controller (wallpaper_controller.py)

Pipeline position (final):

    ... -> renderer.py -> WALLPAPER_CONTROLLER (THIS MODULE) -> 300 seconds
        -> next wallpaper / deletion

This module owns exactly two responsibilities:

    1. talking to the Windows desktop wallpaper API
    2. managing the on-disk lifecycle of *generated* wallpaper files
       (installing them, timing their display, deleting them safely)

It does NOT:
    - fetch NOAA/USGS data
    - compute artistic parameters
    - plan or execute pixel mutations
    - render any image
    - know anything about api_client3.py, parameter_mapper.py,
      region_mapper.py, or mutation_engine.py

renderer.py remains completely independent of this module and of any
Windows wallpaper API. The intended call shape is:

    image_path = renderer.render_wallpaper(live_data, generation_number)
    wallpaper_controller.set_wallpaper(image_path)

============================================================================
WINDOWS MECHANISM USED
============================================================================

Desktop wallpaper get/set is done with the native Win32 call
`SystemParametersInfoW` (via ctypes, no third-party dependency):

    SPI_GETDESKWALLPAPER = 0x0073   -- read the current static wallpaper path
    SPI_SETDESKWALLPAPER = 0x0014   -- set the desktop wallpaper path

This is the same mechanism Windows Settings itself uses under the hood for
a plain "picture" background, and it does not launch any viewer, browser,
or shell process.

============================================================================
SLIDESHOW / NON-STATIC WALLPAPER DETECTION (best effort, documented)
============================================================================

`SPI_GETDESKWALLPAPER` only ever returns a single file path. It cannot
distinguish "the user has one static picture" from "the user has a
slideshow currently showing this particular slide". To avoid silently
treating a slideshow frame as if it were the user's permanent wallpaper,
this module additionally inspects the registry value

    HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Wallpapers
        BackgroundType  (DWORD: 0 = picture, 1 = solid color, 2 = slideshow)

This key is an observed, widely-relied-upon implementation detail of the
Windows Personalization UI, not a documented public Win32 API, so it is
treated as a *heuristic*: if the value is missing/unreadable, this module
assumes a normal static picture rather than guessing "slideshow". If the
value is present and reports slideshow mode, `get_current_wallpaper`
raises `WallpaperControllerError` instead of returning a misleading path.

============================================================================
STATE MANAGEMENT
============================================================================

A single small controller-owned file tracks:

    - the user's original wallpaper path (captured once, never overwritten
      until restored)

This guarantees restoration works cleanly across process restarts or
manual testing (e.g. `python wallpaper_controller.py --restore`).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

try:  # pragma: no cover - only real on Windows
    import winreg
except ImportError:  # not on Windows (e.g. code review/CI on Linux/macOS)
    winreg = None  # type: ignore[assignment]

# ============================================================================
# CONFIGURATION
# ============================================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Matches renderer.py's OUTPUT_DIR. Not imported from renderer.py on
# purpose (renderer.py must stay independent of this module and of any
# Windows API), but the two are intentionally kept in sync as the single
# safety boundary for deletion.
GENERATED_WALLPAPERS_DIR = os.path.join(_THIS_DIR, "generated_wallpapers")

_STATE_FILE = os.path.join(_THIS_DIR, ".wallpaper_original_path")

# Win32 SystemParametersInfo constants (winuser.h)
_SPI_GETDESKWALLPAPER = 0x0073
_SPI_SETDESKWALLPAPER = 0x0014
_SPIF_UPDATEINIFILE = 0x01
_SPIF_SENDWININICHANGE = 0x02
_MAX_PATH = 260  # historical SystemParametersInfoW wallpaper buffer size


class WallpaperControllerError(Exception):
    """Raised for any wallpaper/lifecycle-controller failure.

    Covers: non-Windows platform, invalid/missing files, Windows API
    failures, missing original-wallpaper state, and safety-boundary
    violations on delete. Never raised as a substitute for silently
    ignoring a failure.
    """


class _ControllerState:
    def __init__(self) -> None:
        self.original_wallpaper: Optional[str] = None
        self.active_wallpaper: Optional[str] = None


_state = _ControllerState()


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _require_windows() -> None:
    if os.name != "nt":
        raise WallpaperControllerError(
            "wallpaper_controller requires Windows (os.name == 'nt'); "
            f"got os.name={os.name!r}."
        )


def _validate_existing_file(path: str) -> str:
    """Return an absolute path after confirming it exists and is a
    regular file. Raises WallpaperControllerError otherwise."""
    if not isinstance(path, str) or not path:
        raise WallpaperControllerError(f"image_path must be a non-empty str, got {path!r}")
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise WallpaperControllerError(f"File does not exist: {abs_path}")
    if not os.path.isfile(abs_path):
        raise WallpaperControllerError(f"Not a regular file: {abs_path}")
    return abs_path


def _validate_is_image(abs_path: str) -> None:
    """Best-effort validation that Windows can plausibly use this file
    as a wallpaper image, using PIL (already a project dependency, see
    renderer.py)."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - PIL is a project dep
        raise WallpaperControllerError(
            "Pillow (PIL) is required to validate wallpaper images but is not installed."
        ) from exc

    try:
        with Image.open(abs_path) as img:
            img.verify()
    except Exception as exc:
        raise WallpaperControllerError(
            f"File does not appear to be a valid image Windows can use as wallpaper: "
            f"{abs_path} ({exc})"
        ) from exc


def _is_slideshow_active() -> bool:
    """Best-effort heuristic; see module docstring. Returns False (i.e.
    'assume normal static wallpaper') whenever the registry value cannot
    be read, rather than guessing 'slideshow'."""
    if winreg is None:
        return False
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Wallpapers",
        ) as key:
            value, _type = winreg.QueryValueEx(key, "BackgroundType")
            return int(value) == 2
    except OSError:
        return False


def _call_set_desk_wallpaper(abs_path: str) -> None:
    """Raw Win32 call, no validation/state tracking. Raises
    WallpaperControllerError on failure."""
    import ctypes

    ok = ctypes.windll.user32.SystemParametersInfoW(  # type: ignore[attr-defined]
        _SPI_SETDESKWALLPAPER,
        0,
        abs_path,
        _SPIF_UPDATEINIFILE | _SPIF_SENDWININICHANGE,
    )
    if not ok:
        raise WallpaperControllerError(
            f"SystemParametersInfoW(SPI_SETDESKWALLPAPER) failed for {abs_path}: "
            f"{ctypes.WinError()}"
        )


def _is_within_generated_dir(abs_path: str) -> bool:
    gen_dir = os.path.normcase(os.path.abspath(GENERATED_WALLPAPERS_DIR))
    target = os.path.normcase(abs_path)
    try:
        common = os.path.commonpath([gen_dir, target])
    except ValueError:
        # e.g. different drives on Windows
        return False
    return common == gen_dir


# ============================================================================
# PUBLIC API
# ============================================================================

def set_wallpaper(image_path: str) -> str:
    """
    Set the Windows desktop wallpaper to `image_path`.

    Validates the path exists, is a regular file, and is an image Windows
    can plausibly use, converts it to an absolute path, and calls the
    native Win32 wallpaper-setting API (see module docstring). Never
    opens a viewer or browser.

    Returns:
        The normalized absolute path that was set.

    Raises:
        WallpaperControllerError: not on Windows, invalid/missing file,
        not a valid image, or the Windows API call fails.
    """
    _require_windows()
    abs_path = _validate_existing_file(image_path)
    _validate_is_image(abs_path)
    _call_set_desk_wallpaper(abs_path)
    _state.active_wallpaper = abs_path
    return abs_path


def get_current_wallpaper() -> str:
    """
    Return the current Windows desktop wallpaper as an absolute path.

    Raises:
        WallpaperControllerError: not on Windows, the current
        configuration is detected as a slideshow/theme (not a single
        static file -- see module docstring), the Windows API call
        fails, or no static wallpaper path is reported.
    """
    _require_windows()

    if _is_slideshow_active():
        raise WallpaperControllerError(
            "Current desktop background is a slideshow/theme, not a single "
            "static wallpaper file; refusing to report a single path. "
            "(Detected via the HKCU Explorer\\Wallpapers BackgroundType "
            "heuristic described in this module's docstring.)"
        )

    import ctypes

    buffer = ctypes.create_unicode_buffer(_MAX_PATH)
    ok = ctypes.windll.user32.SystemParametersInfoW(  # type: ignore[attr-defined]
        _SPI_GETDESKWALLPAPER, _MAX_PATH, buffer, 0
    )
    if not ok:
        raise WallpaperControllerError(
            f"SystemParametersInfoW(SPI_GETDESKWALLPAPER) failed: {ctypes.WinError()}"
        )

    path = buffer.value
    if not path:
        raise WallpaperControllerError(
            "Windows reported no static wallpaper path (empty string)."
        )
    return os.path.abspath(path)


def capture_original_wallpaper(force: bool = False) -> str:
    """
    Record the current Windows wallpaper as "the original" to restore
    later. Idempotent: a second call is a no-op and returns the
    previously captured path, unless `force=True`.

    This must be called (directly, or indirectly via
    `show_for_duration`) exactly once before the first Heartbreak
    wallpaper is displayed, so that later Heartbreak wallpapers never
    overwrite the stored original.

    Raises:
        WallpaperControllerError: see `get_current_wallpaper`.
    """
    if not force:
        if _state.original_wallpaper is not None:
            return _state.original_wallpaper
        if os.path.isfile(_STATE_FILE):
            with open(_STATE_FILE, "r", encoding="utf-8") as f:
                path = f.read().strip()
            if path:
                _state.original_wallpaper = path
                return path

    path = get_current_wallpaper()
    _state.original_wallpaper = path
    try:
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            f.write(path)
    except OSError as exc:
        raise WallpaperControllerError(f"Failed to write state file: {exc}")
    
    return path


def restore_original_wallpaper() -> str:
    """
    Restore the previously captured original wallpaper.

    The original file itself is never modified or deleted -- only the
    Windows "current wallpaper" setting is changed back to point at it.
    Controller state is cleared only after the restore succeeds.

    Returns:
        The restored absolute path.

    Raises:
        WallpaperControllerError: no original has been captured yet, or
        the Windows API call fails (in which case state is left intact
        so a caller can retry).
    """
    _require_windows()
    
    original = _state.original_wallpaper
    if not original and os.path.isfile(_STATE_FILE):
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            original = f.read().strip()

    if not original:
        raise WallpaperControllerError(
            "No original wallpaper has been recorded; refusing to guess. "
            "Call capture_original_wallpaper() (or set_wallpaper via "
            "show_for_duration) before restore_original_wallpaper()."
        )

    _call_set_desk_wallpaper(original)  # raises on failure; state left intact
    _state.original_wallpaper = None
    _state.active_wallpaper = None
    
    if os.path.isfile(_STATE_FILE):
        try:
            os.remove(_STATE_FILE)
        except OSError:
            pass

    return original


def delete_wallpaper(image_path: str) -> bool:
    """
    Safely delete a *generated* wallpaper file.

    Safety rules enforced here:
        - the path must resolve inside GENERATED_WALLPAPERS_DIR (never
          an arbitrary path, and never the user's original wallpaper,
          which lives outside that directory by construction)
        - as an extra guard, refuses to delete the wallpaper currently
          tracked as active (`_state.active_wallpaper`) unless the
          caller has already replaced it via a subsequent `set_wallpaper`
          call (which updates `_state.active_wallpaper` to the new
          path) -- pass `force=True` only if you are certain this is
          safe (e.g. cleaning up after a failed/aborted run)

    Args:
        image_path: path to the generated wallpaper file to delete.

    Returns:
        True if a file was deleted, False if the file did not exist
        (treated as a deterministic success/no-op, not an error, since
        the desired end state -- "file is gone" -- already holds).

    Raises:
        WallpaperControllerError: the path falls outside
        GENERATED_WALLPAPERS_DIR, the path is still the active
        wallpaper (without force=True), or deletion fails for any
        other OS-level reason.
    """
    return _delete_wallpaper(image_path, force=False)


def _delete_wallpaper(image_path: str, *, force: bool) -> bool:
    if not isinstance(image_path, str) or not image_path:
        raise WallpaperControllerError(f"image_path must be a non-empty str, got {image_path!r}")
    abs_path = os.path.abspath(image_path)

    if not _is_within_generated_dir(abs_path):
        raise WallpaperControllerError(
            f"Refusing to delete {abs_path}: outside the generated-wallpaper "
            f"safety boundary ({GENERATED_WALLPAPERS_DIR})."
        )

    if (
        not force
        and _state.active_wallpaper is not None
        and os.path.normcase(abs_path) == os.path.normcase(_state.active_wallpaper)
    ):
        raise WallpaperControllerError(
            f"Refusing to delete {abs_path}: it is the currently displayed "
            "wallpaper. Install a replacement with set_wallpaper() first."
        )

    if not os.path.exists(abs_path):
        return False

    try:
        os.remove(abs_path)
    except OSError as exc:
        raise WallpaperControllerError(f"Failed to delete {abs_path}: {exc}") from exc
    return True


def _validate_seconds(seconds) -> None:
    if isinstance(seconds, bool):
        raise WallpaperControllerError(
            f"seconds must be numeric, not bool: {seconds!r}"
        )
    if not isinstance(seconds, (int, float)):
        raise WallpaperControllerError(
            f"seconds must be int or float, got {type(seconds).__name__}: {seconds!r}"
        )
    if seconds <= 0:
        raise WallpaperControllerError(f"seconds must be > 0, got {seconds!r}")


def show_for_duration(image_path: str, seconds) -> None:
    """
    Set `image_path` as the wallpaper and wait `seconds`.

    On the FIRST call in a process, this also captures the user's
    original wallpaper (via `capture_original_wallpaper`) before doing
    anything else, so it can be restored later.

    This function does NOT delete the currently displayed wallpaper merely
    because its timer expired (to safely support future rotation where
    the active wallpaper is kept until replaced).

    Graceful termination (Ctrl+C / KeyboardInterrupt) while waiting:
        1. the wait is interrupted immediately
        2. the original wallpaper is restored (if one was captured)
        3. `image_path` is NOT deleted -- it is left on disk for
           inspection
        4. the KeyboardInterrupt is re-raised so the caller can exit
           cleanly (this function does not call sys.exit itself)

    Args:
        image_path: the generated wallpaper to display.
        seconds: how long to display it. Must be a non-bool int/float > 0.
            The caller supplies this (e.g. 300 for production) -- this
            function never hard-codes a duration.

    Raises:
        WallpaperControllerError: invalid `seconds`, or any failure from
        `set_wallpaper`.
        KeyboardInterrupt: re-raised after restoring the original
        wallpaper, if the wait is interrupted.
    """
    _validate_seconds(seconds)
    capture_original_wallpaper()
    abs_path = set_wallpaper(image_path)
    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        try:
            restore_original_wallpaper()
        except WallpaperControllerError:
            # Best-effort restore; the interrupt itself still takes
            # priority so the process can exit. The generated PNG is
            # left on disk either way (rule 3 above).
            pass
        raise


# ============================================================================
# MANUAL TEST MODE
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        print("--- wallpaper_controller.py manual restore ---")
        try:
            restored = restore_original_wallpaper()
            print(f"[OK] Windows wallpaper restored to: {restored}")
            sys.exit(0)
        except WallpaperControllerError as exc:
            print(f"[FAIL] {exc}")
            sys.exit(1)

    _default_image = os.path.join(
        GENERATED_WALLPAPERS_DIR, "heartbreak_0001.png"
    )
    _image_arg = sys.argv[1] if len(sys.argv) > 1 else _default_image

    print("--- wallpaper_controller.py manual test ---")
    print(f"Target image: {_image_arg}")

    abs_target = os.path.abspath(_image_arg)
    if not os.path.isfile(abs_target):
        print(f"[FAIL] File does not exist or is not a regular file: {abs_target}")
        sys.exit(1)
    print(f"[OK] File exists: {abs_target}")

    try:
        current = get_current_wallpaper()
        print(f"Current/original wallpaper before change: {current}")
    except WallpaperControllerError as exc:
        print(f"[WARN] Could not read current wallpaper: {exc}")

    try:
        capture_original_wallpaper()
        set_path = set_wallpaper(abs_target)
    except WallpaperControllerError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)

    print(f"[OK] Windows wallpaper set to: {set_path}")

    try:
        confirm = get_current_wallpaper()
        print(f"Current wallpaper after change: {confirm}")
        if os.path.normcase(confirm) == os.path.normcase(set_path):
            print("[OK] Windows reports the wallpaper path we just set.")
        else:
            print(
                "[WARN] Windows-reported path differs from what was set "
                "(Windows sometimes stores a transcoded/cached copy)."
            )
    except WallpaperControllerError as exc:
        print(f"[WARN] Could not re-read current wallpaper: {exc}")

    print(
        "\nThis manual test does NOT delete the generated image and does "
        "NOT restore the original wallpaper automatically. Inspect your "
        "desktop now to confirm it changed. "
        "Run `python wallpaper_controller.py --restore` to restore it."
    )
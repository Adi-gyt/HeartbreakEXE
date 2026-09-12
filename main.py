"""
Heartbreak.exe - Main Lifecycle Orchestrator (main.py)
"""

import sys
import time

import api_client3
import renderer
import wallpaper_controller

def run_lifecycle(
    wait_seconds: int = 300,
    fetch_fn=api_client3.get_all_data,
    render_fn=renderer.render_wallpaper,
    set_fn=wallpaper_controller.set_wallpaper,
    delete_fn=wallpaper_controller.delete_wallpaper,
    capture_fn=wallpaper_controller.capture_original_wallpaper,
    restore_fn=wallpaper_controller.restore_original_wallpaper,
    sleep_fn=time.sleep
):
    print("[HEARTBREAK] Starting...")

    # 1. Capture user's original wallpaper
    try:
        capture_fn()
    except Exception as e:
        print(f"[ERROR] Failed to capture original wallpaper: {e}")
        print("[HEARTBREAK] Aborting to ensure we don't trap the user's desktop.")
        sys.exit(1)

    current_wallpaper_path = None
    generation_number = 1

    try:
        while True:
            try:
                # Step 1: Fetch live scientific data
                print("[DATA] Fetching live data...")
                live_data = fetch_fn()

                # Step 2: Generate wallpaper #n
                print(f"[RENDER] Generating wallpaper #{generation_number}...")
                new_wallpaper_path = render_fn(live_data, generation_number)

                # Step 3: Successfully set wallpaper #n
                print(f"[WALLPAPER] Setting wallpaper #{generation_number}...")
                set_fn(new_wallpaper_path)

                # Step 4: Fix State Logic - Update active state IMMEDIATELY upon success
                old_wallpaper_path = current_wallpaper_path
                current_wallpaper_path = new_wallpaper_path

                # Step 5: Cleanup OLD wallpaper now that state reflects the new one
                if old_wallpaper_path:
                    print(f"[CLEANUP] Deleting wallpaper from previous generation: {old_wallpaper_path}")
                    try:
                        delete_fn(old_wallpaper_path)
                    except Exception as e:
                        print(f"[WARN] Failed to delete {old_wallpaper_path}: {e}")

                print("[ROTATION] Complete.")

                # Step 6: Keep active
                print(f"[WAIT] Wallpaper #{generation_number} active for {wait_seconds} seconds...")
                generation_number += 1
                sleep_fn(wait_seconds)

            except Exception as e:
                # Recoverable rotation error (e.g. API timeout, renderer fail, or set_wallpaper fail)
                print(f"[ERROR] Rotation cycle failed: {e}")
                print(f"[WAIT] Retrying in {wait_seconds} seconds... (Current wallpaper kept active)")
                sleep_fn(wait_seconds)

    except KeyboardInterrupt:
        # Step 7: Handle Ctrl+C cleanly
        print("\n[HEARTBREAK] Interrupted by user (Ctrl+C).")
        print("[CLEANUP] Restoring original wallpaper...")
        try:
            restore_fn()
            print("[CLEANUP] Original wallpaper restored.")
        except Exception as e:
            print(f"[ERROR] Failed to restore original wallpaper: {e}")
            
        print("[HEARTBREAK] Exiting cleanly. The last generated wallpaper remains on disk.")
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("[HEARTBREAK] Running in SAFE TEST MODE.")
        
        # Deterministic mock data to avoid network failures during test
        _SAMPLE_LIVE_DATA = {
            "magnitude": 6.5, "depth": 12.0, "latitude": 34.05, "longitude": -118.24,
            "solar_wind_speed": 450.0, "solar_wind_density": 25.0,
            "solar_wind_temperature": 255000.0,
            "bz": -12.5, "by": 10.0, "bt": 15.0, "kp": 4.5,
        }

        class TestState:
            cycles = 0
            set_calls = 0

        def test_fetch():
            return _SAMPLE_LIVE_DATA

        def test_set(path):
            TestState.set_calls += 1
            if TestState.set_calls == 3:
                print("[TEST-STUB] Simulating a set_wallpaper FAILURE on cycle 3.")
                raise Exception("Simulated Windows API failure!")
            return wallpaper_controller.set_wallpaper(path)

        def test_delete(path):
            print(f"[TEST-STUB] delete_wallpaper({path}) WOULD execute safely here.")
            print("[TEST-STUB] Verified: file is preserved to prevent permanent deletion during tests.")

        def test_sleep(seconds):
            TestState.cycles += 1
            if TestState.cycles == 3:
                print("[TEST-STUB] Simulating KeyboardInterrupt (Ctrl+C) to test cleanup...")
                raise KeyboardInterrupt("Simulated Ctrl+C")
            time.sleep(0.1)  # Execute cycles rapidly

        run_lifecycle(
            wait_seconds=2,
            fetch_fn=test_fetch,
            set_fn=test_set,
            delete_fn=test_delete,
            sleep_fn=test_sleep
        )
    else:
        run_lifecycle()
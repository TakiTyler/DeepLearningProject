import os
import sys
import platform

def main():
    print("=== Environment Check ===")
    return_string = ''
    print("Python Version: ", sys.version)
    return_string += "Python Version: " + sys.version + "\n"

    print("Platform: ", platform.platform())
    return_string += "Platform: " + platform.platform() + "\n"

    print("Current Working Directory: ", os.getcwd())
    return_string += "Current Working Directory: " + os.getcwd() + "\n"

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "Not running under Slurm")
    print("Slurm Job ID: ", slurm_job_id)
    return_string += "Slurm Job ID: " + slurm_job_id + "\n"

    print("\n=== File Write Test ===")
    test_dir = "ablation_outputs"
    test_file = os.path.join(test_dir, "test_write_permissions.txt")

    try:
        os.makedirs(test_dir, exist_ok=True)
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("This is a test file to ensure write permissions for model saving.\n")
            f.write(return_string)

        if os.path.exists(test_file):
            print("SUCCESS: Successfully created and wrote to .", test_file)
            print("Model saving operations should work without permission issues.")
        else:
            print("ERROR: File write completed but the file was not found.")
    except Exception as e:
        print("ERROR: Failed to write file. Reason:", e)

if __name__ == "__main__":
    main()

'''20250620 free swap memory, if there's enough RAM to hold data currently in swap space.
Use this for cleaning out swap space.
requires SUDO privileges to run.
'''
import os
import shutil
import subprocess

def run(cmd):
    print(f"> {cmd}")
    return os.system(cmd)

def get_memory_info():
    meminfo = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":")
            meminfo[key.strip()] = int(value.strip().split()[0])  # in KB
    return meminfo

def is_swap_clear_safe(meminfo, buffer_kb=100000):
    """Check if there's enough available RAM to hold swapped data."""
    mem_available = meminfo.get("MemAvailable", 0)
    swap_used = meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0)

    # Allow a buffer to avoid using *all* available memory
    print(f"Available RAM: {mem_available // 1024} MB")
    print(f"Used Swap: {swap_used // 1024} MB")

    if mem_available > (swap_used + buffer_kb):
        return True
    else:
        print("Not enough available RAM to safely clear swap.")
        return False

def clear_memory_and_swap():
    meminfo = get_memory_info()
    print("=== Before ===")
    run("free -m")

    # Clear RAM cache
    run("sudo sync")
    run("sudo sysctl -w vm.drop_caches=3")

    # Clear swap if safe
    if is_swap_clear_safe(meminfo):
        run("sudo swapoff -a")
        run("sudo swapon -a")
    else:
        print("Skipping swap clearing to avoid risk.")

    print("=== After ===")
    run("free -m")

if __name__ == "__main__":
    clear_memory_and_swap()



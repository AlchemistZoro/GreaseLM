import pynvml


# 输入是需要并行计算的gpu数量，输出是符合条件的gpu标号list
def get_gpu_index(gpu_number):
    # Initialize NVML
    pynvml.nvmlInit()

    gpu_index_list = []
    deviceCount = pynvml.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        # Get the GPU utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memrate = meminfo.used/meminfo.total
        if utilization == 0 and memrate < 0.05:
            gpu_index_list.append(i)
            if len(gpu_index_list) == gpu_number:
                return gpu_index_list
    # Shutdown NVML
    pynvml.nvmlShutdown()
    raise ValueError(f"Only {len(gpu_index_list)} gpu are empty. No adequate gpu found !")

# 输入是需要并行计算的gpu数量，输出是符合条件的gpu标号list
def get_avail_gpu_index():
    # Initialize NVML
    pynvml.nvmlInit()

    gpu_index_list = []
    deviceCount = pynvml.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        # Get the GPU utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memrate = meminfo.used/meminfo.total
        if utilization == 0 and memrate < 0.05:
            gpu_index_list.append(i)
    # Shutdown NVML
    pynvml.nvmlShutdown()
    return gpu_index_list


if __name__ == "__main__":
    print(get_gpu_index(1))
    # print(get_gpu_index(2))
    # print(get_gpu_index(8))
    
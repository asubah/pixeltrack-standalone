import subprocess
import itertools
import random

# TODO add cmd arg parser see https://docs.python.org/3/library/argparse.html
# potential args, --verbose, --preheat-gpu, --validation

path = "/home/nfs/asubah/dev/pixeltrack-standalone/"
file = open(path + "autotuning/tunables.csv", "r") 

tunables = {}
tunables_list = []
lbounds = []
ubounds = []
steps = []

for n, line in enumerate(file.readlines()):
    parameters = line.split(',')
    tunables[parameters[0]] = n
    tunables_list.append(parameters[0])
    lbounds.append(int(parameters[1]))
    ubounds.append(int(parameters[2]))
    steps.append(int(parameters[3]))

file.close()

process_path = path + "cuda"

# Heating up the GPU before tuning
subprocess.run([process_path,
    "--runForMinutes", "10",
    "--numberOfThreads", "12",
    "--numberOfStreams", "12"])

configurations = []
ranges = [range(l, u + 1, s) for l, u, s in zip(lbounds, ubounds, steps)]
length = 1
for r in ranges:
    r = list(r)
    length = length * len(r)
    random.shuffle(r)
    configurations.append(r)

print("number of configurations is " + str(length))
# Tuning
configurations = list(itertools.product(*configurations))
random.shuffle(configurations)
for config in configurations:
    kernels_config_path = path + "autotuning/kernel_configs/"
    kernels = tunables_list[:-2]
    # print(kernels)

    # clear previous configs
    # cmd = ['rm', '-f', kernels_config_path + '*']
    # print(cmd)
    # process = subprocess.run(cmd, shell=True)

    # write new configurations
    # TODO write only when the parameter change
    for kernel in kernels:
        # cmd = ['echo', str(config[tunables[kernel]]), '>', kernels_config_path + kernel]
        # print(cmd)
        # subprocess.run(cmd)        
        file = open(kernels_config_path + kernel, 'w')
        file.write(str(config[tunables[kernel]]))
        file.close()
    
    # Validation
    validation = ""
    cmd = [process_path, "--validation"]
    process = subprocess.run(cmd, encoding='UTF-8', capture_output=True)
    if (process.returncode == 0):
        if process.stdout.find("passed validation"):
            validation = "PASSED"
        elif process.stdout.find("failed validation"):
            validation = "FAILED"
        else:
            validation = "ERROR"
    else:
        validation = "ERROR"
        
    # Benchmarking
    time = "NaN"
    throughput = "NaN"
    cpu_efficiency = "NaN"
    status = ""

    cpu_threads = config[tunables["cpu_threads"]]
    gpu_streams = cpu_threads + config[tunables["gpu_streams"]]
    cmd = [process_path,
            "--maxEvents", "10000",
            "--numberOfThreads", str(cpu_threads),
            "--numberOfStreams", str(gpu_streams)]
    process = subprocess.run(cmd, encoding='UTF-8', capture_output=True)

    output = process.stdout
    if (process.returncode == 0):
        # print(output)
        status = "OK"
        output = output.split('\n')[1].split(' ')
        time = output[4]
        throughput = output[7]
        cpu_efficiency = output[13]
    else:
        print(output)
        status = "ERROR"
        
    print(throughput, time, cpu_efficiency, status, validation, config)

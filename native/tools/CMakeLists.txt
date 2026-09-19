add_executable(trt_probe trt_probe.cpp)
target_link_libraries(trt_probe PRIVATE TRT::tensorrt TRT::nvonnxparser)

# 不把 TensorRT 的 DLL 复制到构建目录：nvinfer_11.dll 单个就 383MB，按架构拆分的
# builder resource 每个 113~441MB，复制一份毫无意义。改为让 env.bat 把
# <TRT_ROOT>\bin 挂到 PATH，build.bat 在同一个 shell 里立刻跑一次 trt_probe 验证。

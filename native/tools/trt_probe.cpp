// trt_probe — 在碰真实模型之前，先确认 TensorRT 工具链本身是通的。
//
//   trt_probe                             打印版本（头文件 vs 库，能发现错配）
//   trt_probe inspect <model.onnx>         只解析，列出输入/输出张量
//   trt_probe build   <model.onnx> [选项]   真建一个引擎
//
// 选项
//   --shape NAME=d0xd1x...  优化档位的一条，可重复。动态维必须给全，否则建不出来
//   --workspace MB          工作区上限，默认 1024
//   --save PATH             把序列化引擎写盘
//   --verbose N             解析器日志级别 0..3，默认 1
//
// 没有 --fp16：TensorRT 11 起网络**永远是 strongly-typed** 的，精度由 ONNX 张量
// 类型决定，不再有 BuilderFlag::kFP16。要 FP16 引擎就喂 FP16 的 ONNX。
//
// build 只跑 ONNX parser + builder：它回答的是"经典 TensorRT 到底收不收这张图"，
// 这正是 TRT-RTX 对 gpt_step 回答"不收"的那个问题。

#if defined(_MSC_VER)
// TensorRT 的头文件里仍声明着它自己已弃用的 API，/W4 下会刷满 C4996；我们一个都没用。
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include <NvInfer.h>
#include <NvOnnxParser.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

class Logger final : public nvinfer1::ILogger
{
public:
    explicit Logger(nvinfer1::ILogger::Severity level) noexcept : level_{level} {}

    void log(Severity severity, char const* msg) noexcept override
    {
        if (severity > level_)
        {
            return;
        }
        static char const* const tags[] = {"internal", "error", "warn", "info", "verbose"};
        std::fprintf(stderr, "[TRT %s] %s\n", tags[static_cast<int>(severity)], msg);
    }

private:
    Severity level_;
};

template <typename T>
struct TrtDeleter
{
    void operator()(T* ptr) const noexcept
    {
        delete ptr;
    }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;

double elapsed_ms(std::chrono::steady_clock::time_point const& start)
{
    auto const dt = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double, std::milli>(dt).count();
}

std::string dims_to_string(nvinfer1::Dims const& dims)
{
    std::string out;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (i != 0)
        {
            out += 'x';
        }
        out += std::to_string(dims.d[i]);
    }
    return out.empty() ? std::string{"scalar"} : out;
}

char const* dtype_name(nvinfer1::DataType type)
{
    using DT = nvinfer1::DataType;
    if (type == DT::kFLOAT)
    {
        return "fp32";
    }
    if (type == DT::kHALF)
    {
        return "fp16";
    }
    if (type == DT::kINT8)
    {
        return "int8";
    }
    if (type == DT::kINT32)
    {
        return "int32";
    }
    if (type == DT::kINT64)
    {
        return "int64";
    }
    if (type == DT::kBOOL)
    {
        return "bool";
    }
    return "other";
}

// "24x1x1000x512" -> Dims{24,1,1000,512}. Negative entries are kept, since -1 is
// how a dynamic dimension is spelled.
bool parse_dims(std::string const& text, nvinfer1::Dims* out)
{
    out->nbDims = 0;
    std::size_t start = 0;
    while (start <= text.size())
    {
        std::size_t const sep = text.find('x', start);
        std::string const token =
            text.substr(start, sep == std::string::npos ? std::string::npos : sep - start);
        if (token.empty() || out->nbDims >= nvinfer1::Dims::MAX_DIMS)
        {
            return false;
        }
        out->d[out->nbDims++] = std::atoi(token.c_str());
        if (sep == std::string::npos)
        {
            break;
        }
        start = sep + 1;
    }
    return out->nbDims > 0;
}

struct ProfileEntry
{
    nvinfer1::Dims min{};
    nvinfer1::Dims opt{};
    nvinfer1::Dims max{};
};

// "24x1x1000x512" -> 三档同值；"1x50:1x100:1x256" -> 分别指定 min/opt/max。
bool parse_profile(std::string const& spec, ProfileEntry* out)
{
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (true)
    {
        std::size_t const sep = spec.find(':', start);
        parts.push_back(spec.substr(start, sep == std::string::npos ? std::string::npos : sep - start));
        if (sep == std::string::npos)
        {
            break;
        }
        start = sep + 1;
    }

    if (parts.size() == 1)
    {
        if (!parse_dims(parts[0], &out->min))
        {
            return false;
        }
        out->opt = out->min;
        out->max = out->min;
        return true;
    }
    if (parts.size() == 3)
    {
        return parse_dims(parts[0], &out->min) && parse_dims(parts[1], &out->opt)
            && parse_dims(parts[2], &out->max);
    }
    return false;
}

struct Options
{
    std::string onnx_path;
    std::string save_path;
    std::map<std::string, ProfileEntry> shapes;
    bool inspect_only{false};
    int verbosity{1};
    long workspace_mb{1024};
};

int print_versions()
{
    // 这四个是 extern "C" 全局函数，不在 nvinfer1 命名空间里。
    int const lib_major = getInferLibMajorVersion();
    int const lib_minor = getInferLibMinorVersion();
    int const lib_patch = getInferLibPatchVersion();
    int const lib_build = getInferLibBuildVersion();
    std::printf("TensorRT library : %d.%d.%d.%d\n", lib_major, lib_minor, lib_patch, lib_build);
    std::printf("TensorRT headers : %d.%d.%d.%d\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR,
                NV_TENSORRT_PATCH, NV_TENSORRT_BUILD);
    if (lib_major != NV_TENSORRT_MAJOR)
    {
        std::fprintf(stderr,
                     "!! 主版本不一致：headers=%d 但 lib=%d。链接错版本的库会在运行期以奇怪方式炸。\n",
                     NV_TENSORRT_MAJOR, lib_major);
        return 1;
    }
    return 0;
}

void list_io(nvinfer1::INetworkDefinition const& network, bool* has_dynamic)
{
    *has_dynamic = false;
    for (int i = 0; i < network.getNbInputs(); ++i)
    {
        nvinfer1::ITensor const* t = network.getInput(i);
        nvinfer1::Dims const dims = t->getDimensions();
        for (int d = 0; d < dims.nbDims; ++d)
        {
            if (dims.d[d] < 0)
            {
                *has_dynamic = true;
            }
        }
        std::printf("  in  %-18s %-6s %s\n", t->getName(), dtype_name(t->getType()),
                    dims_to_string(dims).c_str());
    }
    for (int i = 0; i < network.getNbOutputs(); ++i)
    {
        nvinfer1::ITensor const* t = network.getOutput(i);
        std::printf("  out %-18s %-6s %s\n", t->getName(), dtype_name(t->getType()),
                    dims_to_string(t->getDimensions()).c_str());
    }
}

int run(Options const& opt)
{
    nvinfer1::ILogger::Severity const level = opt.verbosity >= 2 ? nvinfer1::ILogger::Severity::kVERBOSE
                                                                : nvinfer1::ILogger::Severity::kWARNING;
    Logger logger{level};

    TrtPtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
    if (!builder)
    {
        std::fprintf(stderr, "createInferBuilder 失败\n");
        return 2;
    }

    // TensorRT 11 去掉了 kEXPLICIT_BATCH —— 显式 batch 成了唯一模式，网络也永远
    // 是 strongly-typed，所以精度来自 ONNX 张量类型而不是 builder flag。
    TrtPtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0)};
    if (!network)
    {
        std::fprintf(stderr, "createNetworkV2 失败\n");
        return 2;
    }

    TrtPtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger)};
    if (!parser)
    {
        std::fprintf(stderr, "createParser 失败\n");
        return 2;
    }

    std::printf("parsing %s\n", opt.onnx_path.c_str());
    auto const t_parse = std::chrono::steady_clock::now();
    bool const parsed = parser->parseFromFile(opt.onnx_path.c_str(), opt.verbosity);
    double const parse_ms = elapsed_ms(t_parse);
    int const nerrors = parser->getNbErrors();
    for (int i = 0; i < nerrors; ++i)
    {
        std::fprintf(stderr, "  parse error %d: %s\n", i, parser->getError(i)->desc());
    }
    if (!parsed)
    {
        std::fprintf(stderr, "解析失败（%.0f ms，%d 条错误）\n", parse_ms, nerrors);
        return 3;
    }
    std::printf("parsed in %.0f ms\n", parse_ms);

    bool has_dynamic = false;
    list_io(*network, &has_dynamic);

    if (opt.inspect_only)
    {
        std::printf("dynamic inputs: %s\n", has_dynamic ? "yes" : "no");
        return 0;
    }

    TrtPtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (!config)
    {
        std::fprintf(stderr, "createBuilderConfig 失败\n");
        return 2;
    }

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               static_cast<std::size_t>(opt.workspace_mb) << 20);

    if (!opt.shapes.empty())
    {
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        if (!profile)
        {
            std::fprintf(stderr, "createOptimizationProfile 失败\n");
            return 2;
        }
        for (auto const& [name, entry] : opt.shapes)
        {
            if (!profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN, entry.min)
                || !profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT, entry.opt)
                || !profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX, entry.max))
            {
                std::fprintf(stderr, "setDimensions 拒绝 %s\n", name.c_str());
                return 4;
            }
            std::printf("  profile %-18s min=%s opt=%s max=%s\n", name.c_str(),
                        dims_to_string(entry.min).c_str(), dims_to_string(entry.opt).c_str(),
                        dims_to_string(entry.max).c_str());
        }
        if (config->addOptimizationProfile(profile) < 0)
        {
            std::fprintf(stderr, "addOptimizationProfile 失败\n");
            return 4;
        }
    }
    else if (has_dynamic)
    {
        std::fprintf(stderr, "图里有动态维，但没有给 --shape：TensorRT 建不出来。\n");
        return 4;
    }

    std::printf("building engine ...\n");
    auto const t_build = std::chrono::steady_clock::now();
    TrtPtr<nvinfer1::IHostMemory> engine{builder->buildSerializedNetwork(*network, *config)};
    double const build_ms = elapsed_ms(t_build);
    if (!engine)
    {
        std::fprintf(stderr, "buildSerializedNetwork 返回空（%.0f ms）：建引擎失败，看上面的 TRT 诊断\n",
                     build_ms);
        return 5;
    }

    std::printf("engine built in %.0f ms, %.1f MB\n", build_ms, engine->size() / 1048576.0);
    if (!opt.save_path.empty())
    {
        std::ofstream out{opt.save_path, std::ios::binary};
        out.write(static_cast<char const*>(engine->data()), static_cast<std::streamsize>(engine->size()));
        if (!out)
        {
            std::fprintf(stderr, "写 %s 失败\n", opt.save_path.c_str());
            return 6;
        }
        std::printf("saved %s\n", opt.save_path.c_str());
    }
    return 0;
}

void print_usage()
{
    std::fprintf(stderr,
                 "usage:\n"
                 "  trt_probe\n"
                 "  trt_probe inspect <model.onnx> [--verbose N]\n"
                 "  trt_probe build   <model.onnx> [--shape NAME=d0xd1x...|min:opt:max]... "
                 "[--workspace MB] [--save PATH] [--verbose N]\n");
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        return print_versions();
    }

    std::string const command = argv[1];
    if (command == "--help" || command == "-h")
    {
        print_usage();
        return 0;
    }

    Options opt;
    opt.inspect_only = (command == "inspect");
    if (!opt.inspect_only && command != "build")
    {
        print_usage();
        return 1;
    }

    for (int i = 2; i < argc; ++i)
    {
        std::string const arg = argv[i];
        auto const next = [&](char const* what) -> char const* {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "%s 缺少参数\n", what);
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--verbose")
        {
            opt.verbosity = std::atoi(next("--verbose"));
        }
        else if (arg == "--workspace")
        {
            opt.workspace_mb = std::atol(next("--workspace"));
        }
        else if (arg == "--save")
        {
            opt.save_path = next("--save");
        }
        else if (arg == "--shape")
        {
            std::string const spec = next("--shape");
            std::size_t const eq = spec.find('=');
            if (eq == std::string::npos)
            {
                std::fprintf(stderr, "--shape 需要 NAME=d0xd1x... 或 NAME=min:opt:max，收到: %s\n", spec.c_str());
                return 1;
            }
            ProfileEntry entry{};
            if (!parse_profile(spec.substr(eq + 1), &entry))
            {
                std::fprintf(stderr, "无法解析形状: %s\n", spec.c_str());
                return 1;
            }
            opt.shapes.emplace(spec.substr(0, eq), entry);
        }
        else if (opt.onnx_path.empty())
        {
            opt.onnx_path = arg;
        }
        else
        {
            std::fprintf(stderr, "多余参数: %s\n", arg.c_str());
            return 1;
        }
    }

    if (opt.onnx_path.empty())
    {
        print_usage();
        return 1;
    }
    return run(opt);
}

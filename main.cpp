#include <thread>
#include <cmath>
#include <omp.h>
#include <atomic>
#include <direct.h>
#include <array>
#include <vector>
#include <algorithm>

#include "progress.h"

#define SINGLE_THREADED() false

template <size_t INPUT_DIMENSIONS, size_t STEP_SIZE, size_t KEEP_COUNT>
struct Optimizer_Base
{
    static const size_t c_INPUT_DIMENSIONS = 1;
    using TInput = std::array<float, c_INPUT_DIMENSIONS>;

    inline static const size_t c_KEEP_COUNT = KEEP_COUNT;
    inline static const size_t c_STEP_SIZE = STEP_SIZE;

    inline static const float s_startf = 0.0f;
    inline static const float s_endf = 1.0f;

    inline static uint32_t s_start = *reinterpret_cast<const uint32_t*>(&s_startf);
    inline static uint32_t s_end = *reinterpret_cast<const uint32_t*>(&s_endf);

    static void InitInput(TInput& input)
    {
        for (float& f : input)
            f = 0.0f;
    }

    static float GetInputPercent(const TInput& input)
    {
        float ret = 0.0f;
        float divider = 0.01f;
        for (const float& f : input)
        {
            uint32_t u = *reinterpret_cast<const uint32_t*>(&f);
            ret += (float(u - s_start) / float(s_end - s_start)) / divider;

            divider *= 100.0f;
        }
        return ret;
    }

#if 0
    // Note: this is A LOT slower than the #else way
    static bool AdvanceInput(TInput& input)
    {
        for (float& f : input)
        {
            f = std::nextafter(f, 1.0f);
            if (f < 1.0f)
                return true;
            f = 0.0f;
        }
        return false;
    }

    static bool AdvanceInput(TInput& input, int count)
    {
        for (int i = 0; i < count * c_STEP_SIZE; ++i)
        {
            if (!AdvanceInput(input))
                return false;
        }
        return true;
    }
#else

    static bool AdvanceInput(TInput& input, int count)
    {
        count *= c_STEP_SIZE;
        for (float& f : input)
        {
            uint32_t u = *reinterpret_cast<uint32_t*>(&f);
            u += count;
            if (u < s_end)
            {
                f = *reinterpret_cast<float*>(&u);
                return true;
            }

            u = s_start;
            count = 1 + s_end - u;

            f = *reinterpret_cast<float*>(&u);
        }
        return false;
    }

#endif

    struct PerThreadData
    {
        struct Result
        {
            typename TInput input;
            float score;
        };

        PerThreadData()
        {
            for (Result& result : results)
                result.score = FLT_MAX;
        }

        void ProcessResult(const TInput& input, float score)
        {
            for (Result& oldResult : results)
            {
                if (score < oldResult.score)
                {
                    oldResult = { input, score };
                    return;
                }
            }
        }

        std::array<Result, c_KEEP_COUNT> results;
    };
};

template <size_t STEP_SIZE, size_t KEEP_COUNT>
struct Optimize_1D : public Optimizer_Base<1, STEP_SIZE, KEEP_COUNT>
{
    using TBase = Optimizer_Base<1, STEP_SIZE, KEEP_COUNT>;
    using TInput = TBase::TInput;

    static float Score(const TInput& input)
    {
        return std::abs(input[0] - 0.5f);
    }
};

template <typename Optimizer>
void Optimize(const char* baseName)
{
    // use all the threads we can
    unsigned int numThreads = std::thread::hardware_concurrency();
    numThreads = std::max(numThreads, 1u);
    #if SINGLE_THREADED()
        numThreads = 1;
    #endif
    omp_set_num_threads(numThreads);

    printf("%s - %u threads...\n", baseName, numThreads);

    ProgressContext progress;

    std::vector<Optimizer::PerThreadData> threadData(numThreads);

    #pragma omp parallel for
    for (int i = 0; i < (int)numThreads; ++i)
    {
        bool report = (i == 0);

        typename Optimizer::TInput x;
        typename Optimizer::InitInput(x);
        typename Optimizer::AdvanceInput(x, i);

        do
        {
            float y = Optimizer::Score(x);
            threadData[i].ProcessResult(x, y);

            if (report)
            {
                float f = typename Optimizer::GetInputPercent(x);
                progress.Report(int(f * 10.0f), 1000);
            }
        }
        while (Optimizer::AdvanceInput(x, numThreads));
    }
    progress.Report(1, 1);

    // Collect the N winners from the multiple threads
    typename Optimizer::PerThreadData finalData;
    for (typename Optimizer::PerThreadData& data : threadData)
    {
        for (const auto& result : data.results)
            finalData.ProcessResult(result.input, result.score);
    }

    // sort results
    std::sort(
        finalData.results.begin(),
        finalData.results.end(),
        [](const auto& a, const auto& b) { return a.score < b.score; }
    );

    // Write results csv
    char fileName[1024];
    sprintf_s(fileName, "out/%s.csv", baseName);
    FILE* file = nullptr;
    fopen_s(&file, fileName, "w+t");

    // write header
    for (int i = 0; i < Optimizer::c_INPUT_DIMENSIONS; ++i)
        fprintf(file, "\"x%i\",", i);
    fprintf(file, "\"score\"\n");

    // write data rows
    for (const auto& result : finalData.results)
    {
        if (result.score == FLT_MAX)
            continue;

        for (int i = 0; i < Optimizer::c_INPUT_DIMENSIONS; ++i)
            fprintf(file, "\"%f\",", result.input[i]);
        fprintf(file, "\"%f\"\n", result.score);
    }

    fclose(file);
}

int main(int argc, char** argv)
{
    _mkdir("out");

    Optimize<Optimize_1D<1, 5>>("test");

    return 0;
}

/*
TODO:
* come up with compelling things to optimize for and see how they do
 * 1D - 1d blue noise?
 * 2D - IGN? blue noise? plus shape sampling?
 * 3D - spatiotemporal something?

Notes:
* uses as many threads as available.
* you can have it step by more than 1 to take sparser samples and go faster.


*/

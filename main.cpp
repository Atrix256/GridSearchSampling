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
    static const size_t c_INPUT_DIMENSIONS = INPUT_DIMENSIONS;
    using TInput = std::array<float, c_INPUT_DIMENSIONS>;

    inline static const size_t c_KEEP_COUNT = KEEP_COUNT;
    inline static const size_t c_STEP_SIZE = STEP_SIZE;

    inline static const float s_endf = 1.0f;
    inline static uint32_t s_end = *reinterpret_cast<const uint32_t*>(&s_endf);

    static float GetInputPercent(const TInput& input)
    {
        float f = input[input.size() - 1];
        uint32_t u = *reinterpret_cast<const uint32_t*>(&f);
        return 100.0f * (float(u) / float(s_end));


        /*
        float ret = 0.0f;
        float divider = 0.01f;
        for (int i = (int)input.size() - 1; i >= 0; --i)
        {
            uint32_t u = *reinterpret_cast<const uint32_t*>(&input[i]);
            ret += (float(u - s_start) / float(s_end - s_start)) / divider;
            divider *= 100.0f;
        }
        return ret;
        */
    }

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

        inline void ProcessResult(const TInput& input, float score)
        {
            if (c_KEEP_COUNT == 1)
            {
                if (score < results[0].score)
                {
                    results[0].input = input;
                    results[0].score = score;
                }
            }
            else
            {
                // Index 0 is always the highest score.
                // Keep this value only if it's less than the highest score.
                // Then, find and swap to the new highest score

                if (results[0].score > score)
                {
                    results[0] = { input, score };
                    int largestScoreIndex = 0;
                    float largestScore = score;
                    for (int i = 1; i < results.size(); ++i)
                    {
                        if (results[i].score > largestScore)
                        {
                            largestScoreIndex = i;
                            largestScore = results[i].score;
                        }
                    }
                    if (largestScoreIndex > 0)
                    {
                        Result temp = results[0];
                        results[0] = results[largestScoreIndex];
                        results[largestScoreIndex] = temp;
                    }
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

    static inline float Score(const TInput& input)
    {
        return std::abs(input[0] - 0.5f);
    }
};

template <size_t STEP_SIZE, size_t KEEP_COUNT>
struct Optimize_2D : public Optimizer_Base<2, STEP_SIZE, KEEP_COUNT>
{
    using TBase = Optimizer_Base<2, STEP_SIZE, KEEP_COUNT>;
    using TInput = TBase::TInput;

    static inline float Score(const TInput& input)
    {
        return std::abs((input[0] * input[1]) - 0.618f);
    }
};

template <typename Optimizer, size_t INDEX>
void IterateInput(typename Optimizer::TInput& input, float min, float max, typename Optimizer::PerThreadData& threadData)
{
    bool report = (INDEX == 0 && min == 0.0f);

    ProgressContext progress;

    input[INDEX] = min;
    while (input[INDEX] < max)
    {
        if (INDEX < Optimizer::c_INPUT_DIMENSIONS - 1)
        {
            IterateInput<Optimizer, INDEX + 1>(input, 0.0f, 1.0f, threadData);
        }
        else
        {
            float y = Optimizer::Score(input);
            threadData.ProcessResult(input, y);
        }

        uint32_t u = *reinterpret_cast<uint32_t*>(&input[INDEX]);
        u += Optimizer::c_STEP_SIZE;
        input[INDEX] = *reinterpret_cast<float*>(&u);

        if (report)
        {
            float f = typename Optimizer::GetInputPercent(input);
            progress.Report(int(f * 100.0f), 10000);
        }
    }

    if (report)
        progress.Report(1, 1);
}

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

    std::vector<Optimizer::PerThreadData> threadData(numThreads);

    #pragma omp parallel for
    for (int i = 0; i < (int)numThreads; ++i)
    {
        typename Optimizer::TInput x;

        float min = float(i) / float(numThreads);
        float max = float(i + 1) / float(numThreads);

        IterateInput<Optimizer, 0>(x, min, max, threadData[i]);
    }

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

    //Optimize<Optimize_1D<1, 5>>("test1d");
    Optimize<Optimize_2D<65536, 1>>("test2d");

    return 0;
}

/*
TODO:
! in the 2D case, if you think of it as a double for loop, you are incrementing one by the step size, but not the other!
 * maybe actually do a double for loop? have the inner most for loop divied up

* come up with compelling things to optimize for and see how they do
 * 1D - 1d blue noise?
 * 2D - IGN? blue noise? plus shape sampling?
 * 3D - spatiotemporal something?

Notes:
* uses as many threads as available.
* you can have it step by more than 1 to take sparser samples and go faster.


*/

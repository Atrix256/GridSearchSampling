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

static const float c_goldenRatio = 1.618033988749894f;
static const float c_goldenRatioConjugate = 0.618033988749894f;

template <typename T>
T fract(const T& t)
{
    return t - floor(t);
}

template <size_t INPUT_DIMENSIONS, size_t STEP_SIZE, size_t KEEP_COUNT>
struct Optimizer_Base
{
    static const size_t c_INPUT_DIMENSIONS = INPUT_DIMENSIONS;
    using TInput = std::array<float, c_INPUT_DIMENSIONS>;

    inline static const size_t c_KEEP_COUNT = KEEP_COUNT;
    inline static const size_t c_STEP_SIZE = STEP_SIZE;

    inline static const float s_endf = 1.0f;
    inline static uint32_t s_end = *reinterpret_cast<const uint32_t*>(&s_endf);

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
            // special case known at compile time, for if we are only keeping one
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
struct Optimize_2D_Coirrational : public Optimizer_Base<2, STEP_SIZE, KEEP_COUNT>
{
    using TBase = Optimizer_Base<2, STEP_SIZE, KEEP_COUNT>;
    using TInput = TBase::TInput;

    // Goal: find two values which are maximally irrational from each other and the golden ratio
    static inline float Score(const TInput& input)
    {
        if (input[0] < 0.0001f || input[1] < 0.0001f)
            return FLT_MAX;

        float error1 = std::abs(fract(input[0] / input[1]) - c_goldenRatioConjugate);
        float error2 = std::abs(fract(input[1] / input[0]) - c_goldenRatioConjugate);
        float error3 = std::abs(fract(input[0] / c_goldenRatioConjugate) - c_goldenRatioConjugate);
        float error4 = std::abs(fract(input[1] / c_goldenRatioConjugate) - c_goldenRatioConjugate);

        return sqrt(error1 * error1 + error2 * error2 + error3 * error3 + error4 * error4);
    }
};

template <size_t STEP_SIZE, size_t KEEP_COUNT>
struct Optimize_3D : public Optimizer_Base<3, STEP_SIZE, KEEP_COUNT>
{
    using TBase = Optimizer_Base<3, STEP_SIZE, KEEP_COUNT>;
    using TInput = TBase::TInput;

    static inline float Score(const TInput& input)
    {
        return std::abs(fract(input[0] * input[1] * input[2]) - 0.618f);
    }
};

inline bool AdvanceFloat(float& f, uint32_t stepCount, float max)
{
    uint32_t maxu = *reinterpret_cast<uint32_t*>(&max);

    uint32_t u = *reinterpret_cast<uint32_t*>(&f);
    u += stepCount;
    bool ret = u < maxu;
    u %= maxu;
    f = *reinterpret_cast<float*>(&u);

    return ret;
}

template <typename Optimizer>
void IterateInput(typename Optimizer::TInput& input, float min, float max, typename Optimizer::PerThreadData& threadData, int threadId)
{
    ProgressContext progress;
    bool report = (threadId == 0);

    uint32_t minu = *reinterpret_cast<uint32_t*>(&min);
    uint32_t maxu = *reinterpret_cast<uint32_t*>(&max);

    // initialize the input
    input[0] = min;
    for (int i = 1; i < Optimizer::c_INPUT_DIMENSIONS; ++i)
        input[i] = 0.0f;

    // loop until we are done with this slice of input
    while (true)
    {
        // process this input value
        float y = Optimizer::Score(input);
        threadData.ProcessResult(input, y);

        // advance to the next input value if we can
        bool couldIterate = false;
        for (int i = Optimizer::c_INPUT_DIMENSIONS - 1; i >= 0; --i)
        {
            if (AdvanceFloat(input[i], Optimizer::c_STEP_SIZE, (i == 0) ? max : 1.0f))
            {
                couldIterate = true;
                break;
            }
        }

        // exit if we are done
        if (!couldIterate)
            break;

        // report progress
        if (report)
        {
            uint32_t currentu = *reinterpret_cast<uint32_t*>(&input[0]);
            float f = float(currentu - minu) / float(maxu - minu);
            progress.Report(int(f * 10000.0f), 10000);
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

        uint32_t minu = uint32_t(float(Optimizer::s_end) * float(i) / float(numThreads));
        uint32_t maxu = uint32_t(float(Optimizer::s_end) * float(i+1) / float(numThreads));
        float min = *reinterpret_cast<float*>(&minu);
        float max = *reinterpret_cast<float*>(&maxu);

        IterateInput<Optimizer>(x, min, max, threadData[i], i);
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
    for (int i = 0; i < Optimizer::c_INPUT_DIMENSIONS; ++i)
        fprintf(file, "\"x%i as uint32\",", i);
    fprintf(file, "\"score\"\n");
    fprintf(file, "\"score as uint32\"\n");

    // write data rows
    for (const auto& result : finalData.results)
    {
        if (result.score == FLT_MAX)
            continue;

        for (int i = 0; i < Optimizer::c_INPUT_DIMENSIONS; ++i)
            fprintf(file, "\"%f\",", result.input[i]);
        for (int i = 0; i < Optimizer::c_INPUT_DIMENSIONS; ++i)
            fprintf(file, "\"%u\",", *reinterpret_cast<const uint32_t*>(&result.input[i]));
        fprintf(file, "\"%f\"\n", result.score);
        fprintf(file, "\"%u\"\n", *reinterpret_cast<const uint32_t*>(&result.score));
    }

    fclose(file);
}

int main(int argc, char** argv)
{
    _mkdir("out");

    //Optimize<Optimize_1D<1, 5>>("test1d");
    Optimize<Optimize_2D_Coirrational<1024 * 16, 5>>("coirrational");
    //Optimize<Optimize_3D<1024 * 256, 5>>("test3d");

    return 0;
}

/*
TODO:

* have an option for gradient descent on the winners
* is there a way to optimize situations where x1 and x2 are interchangeable, so it decreases the search space?
 * yeah. if they are interchangeable, inner loops start at the outer loop current value i think?
 * it will also help find more unique solutions, which is nice

* come up with compelling things to optimize for and see how they do
 * 1D - 1d blue noise?
 * 2D - IGN? blue noise? plus shape sampling?
 * 3D - spatiotemporal something?

Notes:
* uses as many threads as available.
* you can have it step by more than 1 to take sparser samples and go faster.
* note how squaring error terms that are summed together promotes error being balanced across those terms, instead of lumped into one term.

*/

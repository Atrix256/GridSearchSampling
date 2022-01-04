#pragma once

#include <string>
#include <chrono>

struct ProgressContext
{
    ProgressContext()
    {
        m_timeStart = std::chrono::high_resolution_clock::now();
        m_lastPercent = -1;
        m_lastBufferLength = 0;
    }

    int m_lastPercent;
    std::chrono::high_resolution_clock::time_point m_timeStart;
    int m_lastBufferLength;

    inline static std::string MakeDurationString(float durationInSeconds)
    {
        std::string ret;

        static const float c_oneMinute = 60.0f;
        static const float c_oneHour = c_oneMinute * 60.0f;

        int hours = int(durationInSeconds / c_oneHour);
        durationInSeconds -= float(hours) * c_oneHour;

        int minutes = int(durationInSeconds / c_oneMinute);
        durationInSeconds -= float(minutes) * c_oneMinute;

        int seconds = int(durationInSeconds);

        char buffer[1024];
        if (hours < 10)
            sprintf_s(buffer, "0%i:", hours);
        else
            sprintf_s(buffer, "%i:", hours);
        ret = buffer;

        if (minutes < 10)
            sprintf_s(buffer, "0%i:", minutes);
        else
            sprintf_s(buffer, "%i:", minutes);
        ret += buffer;

        if (seconds < 10)
            sprintf_s(buffer, "0%i", seconds);
        else
            sprintf_s(buffer, "%i", seconds);
        ret += buffer;

        return ret;
    }

    inline bool Report(size_t count, size_t total)
    {
        float percentF = 100.0f * float(double(count) / double(total));
        int percentx10 = int(percentF * 10.0f);

        if (m_lastPercent == percentx10)
            return false;
        m_lastPercent = percentx10;

        // get time since start
        std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> timeSpan = std::chrono::duration_cast<std::chrono::duration<double>>(now - m_timeStart);

        std::string elapsed = MakeDurationString((float)timeSpan.count());

        float estimateMultiplier = std::max(100.0f / percentF, 1.0f);
        std::string estimate = MakeDurationString(estimateMultiplier * (float)timeSpan.count());

        // make the message
        char buffer[1024];
        if (count == total)
            sprintf_s(buffer, "\r100%%  elapsed %s", elapsed.c_str());
        else
            sprintf_s(buffer, "\r%0.1f%%  elapsed %s  estimated %s", float(percentx10) / 10.0f, elapsed.c_str(), estimate.c_str());

        int newBufferLength = (int)strlen(buffer);
        int newBufferLengthCopy = newBufferLength;

        // pad with spaces to erase whatever may be there from before
        while (newBufferLength < m_lastBufferLength)
        {
            buffer[newBufferLength] = ' ';
            newBufferLength++;
        }
        buffer[newBufferLength] = 0;
        if (count == total)
            strcat_s(buffer, "\n");

        // show the message
        printf("%s", buffer);
        m_lastBufferLength = newBufferLengthCopy;

        return true;
    }
};

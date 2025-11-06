#pragma once

#include <algorithm>
#include <cctype>
#include <functional>
#include <locale>

/**
 * @brief Trims whitespace from the start of a string (in place).
 *
 * This function removes all leading whitespace characters from the input
 * string.
 *
 * @param s The input string to trim.
 */
constexpr void TrimL(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
}

/**
 * @brief Trims whitespace from the end of a string (in place).
 *
 * This function removes all trailing whitespace characters from the input
 * string.
 *
 * @param s The input string to trim.
 */
constexpr void TrimR(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](unsigned char ch) { return !std::isspace(ch); })
              .base(),
          s.end());
}

/**
 * @brief Trims whitespace from both ends of a string (in place).
 *
 * This function removes all leading and trailing whitespace characters from the
 * input string.
 *
 * @param s The input string to trim.
 */
constexpr void Trim(std::string &s) {
  TrimR(s);
  TrimL(s);
}

/**
 * @brief Trims whitespace from the start of a string (copying).
 *
 * This function returns a new string with all leading whitespace characters
 * removed.
 *
 * @param s The input string to trim.
 * @return A new string with leading whitespace removed.
 */
constexpr std::string TrimLCopy(std::string s) {
  TrimL(s);
  return s;
}

/**
 * @brief Trims whitespace from the end of a string (copying).
 *
 * This function returns a new string with all trailing whitespace characters
 * removed.
 *
 * @param s The input string to trim.
 * @return A new string with trailing whitespace removed.
 */
constexpr std::string TrimRCopy(std::string s) {
  TrimR(s);
  return s;
}

/**
 * @brief Trims whitespace from both ends of a string (copying).
 *
 * This function returns a new string with all leading and trailing whitespace
 * characters removed.
 *
 * @param s The input string to trim.
 * @return A new string with leading and trailing whitespace removed.
 */
constexpr std::string TrimCopy(std::string s) {
  Trim(s);
  return s;
}

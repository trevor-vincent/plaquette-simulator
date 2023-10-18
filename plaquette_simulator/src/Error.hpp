#pragma once

#include <exception>
#include <iostream>
#include <sstream>

/**
 * @brief Macro that throws `%Exception` with given message.
 *
 * @param message string literal describing error
 */
#define PLAQUETTE_ABORT(message)                                               \
    Plaquette::Abort(message, __FILE__, __LINE__, __func__)
/**
 * @brief Macro that throws `%Exception` if expression evaluates to true.
 *
 * @param expression an expression
 * @param message string literal describing error
 */
#define PLAQUETTE_ABORT_IF(expression, message)                                \
    if ((expression)) {                                                        \
        PLAQUETTE_ABORT(message);                                              \
    }
/**
 * @brief Macro that throws `%Exception` with error message if expression
 * evaluates to false.
 *
 * @param expression an expression
 * @param message string literal describing error
 */
#define PLAQUETTE_ABORT_IF_NOT(expression, message)                            \
    if (!(expression)) {                                                       \
        PLAQUETTE_ABORT(message);                                              \
    }

/**
 * @brief Macro that throws `%Exception` with the given expression and source
 * location if expression evaluates to false.
 *
 * @param expression an expression
 */
#define PLAQUETTE_ASSERT(expression)                                           \
    PLAQUETTE_ABORT_IF_NOT(expression, "Assertion failed: " #expression)

namespace Plaquette {

/**
 * @brief `%Exception` is the general exception thrown by Plaquette for runtime
 * errors.
 *
 */
class Exception : public std::exception {
  public:
    /**
     * @brief Constructs a new `%Exception` exception.
     *
     * @param err_msg Error message explaining the exception condition.
     */
    explicit Exception(const std::string &err_msg) noexcept
        : err_msg(err_msg) {}

    /**
     * @brief Destroys the `%Exception` object.
     */
    virtual ~Exception() = default;

    /**
     * @brief Returns a string containing the exception message. Overrides
     *        the `std::exception` method.
     *
     * @return Exception message.
     */
    const char *what() const noexcept { return err_msg.c_str(); }

  private:
    std::string err_msg;
};

/**
 * @brief Throws an `%Exception` with the given error message.
 *
 * This function should not be called directly - use one of the
 * `PLAQUETTE_ASSERT()` or `PLAQUETTE_ABORT()` macros, which provide the source
 * location at compile time.
 *
 * @param message string literal describing the error
 * @param file_name source file where error occured
 * @param line line of source file
 * @param function_name function in which error occured
 */
[[noreturn]] inline void Abort(const char *message, const char *file_name,
                               int line, const char *function_name) {
    std::stringstream err_msg;
    err_msg << "[" << file_name << "][Line:" << line
            << "][Method:" << function_name
            << "]: Error in Plaquette: " << message;
    throw Exception(err_msg.str());
}

}; // namespace Plaquette

#include <fasttext/include/fasttext.h>
#include <sstream>

#include "predictions.h"

BEGIN_EXTERN_C()

// FastText_Predict_t FastText_Analogy(const FastText_Handle_t handle, FastText_String_t word1, FastText_String_t word2,
//                                     FastText_String_t word3, uint32_t k)
// {
//     const auto model = reinterpret_cast<fasttext::FastText *>(handle);
//     Predictions predictions =
//         model->getAnalogies(k, std::string(word1.data, word1.size), std::string(word2.data, word2.size),
//                             std::string(word3.data, word3.size));

//     auto vec = new Predictions(std::move(predictions));

//     return FastText_Predict_t{
//         vec->size(),
//         (void *)vec,
//     };
// }

FastText_FloatVector_t FastText_Wordvec(const FastText_Handle_t handle, FastText_String_t word)
{
    const auto model = reinterpret_cast<fasttext::FastText *>(handle);
    auto vec = new fasttext::Vector(std::move(model->getWordVector(std::string_view(word.data, word.size))));

    return FastText_FloatVector_t{
        vec->data(),
        (void *)vec,
        (size_t)vec->size(),
    };
}

FastText_FloatVector_t FastText_Sentencevec(const FastText_Handle_t handle, FastText_String_t sentence)
{
    const auto model = reinterpret_cast<fasttext::FastText *>(handle);
    std::stringstream ss(sentence.data);
    auto vec = new fasttext::Vector(std::move(model->getSentenceVector(ss)));

    return FastText_FloatVector_t{
        vec->data(),
        (void *)vec,
        (size_t)vec->size(),
    };
}

void FastText_FreeFloatVector(FastText_FloatVector_t vector)
{
    auto vec = reinterpret_cast<fasttext::Vector *>(vector.handle);
    delete vec;
}

END_EXTERN_C()

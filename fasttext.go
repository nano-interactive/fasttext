package fasttext

// #cgo CXXFLAGS: -I${SRCDIR}/fasttext/include -I${SRCDIR} -std=c++17 -Ofast -fPIC -pthread -Wno-defaulted-function-deleted
// #include <stdio.h>
// #include <stdlib.h>
// #include <stdint.h>
// #include "predictions.h"
import "C"

import (
	"errors"
	"runtime"
	"unsafe"
)

var (
	ErrPredictionFailed = errors.New("prediction failed")
	ErrNoPredictions    = errors.New("no predictions")
)

// A model object. Effectively a wrapper
// around the C fasttext handle
type (
	Model struct {
		p C.FastText_Handle_t
	}

	ModelOpenError string
)

func (e ModelOpenError) Error() string {
	return string(e)
}

// Open opens a model from a path and returns a model
// object
func Open(path string) (Model, error) {
	var pinner runtime.Pinner
	defer pinner.Unpin()

	cStrPath := cStr(path)
	pinner.Pin(cStrPath)
	result := C.FastText_NewHandle(C.FastText_String_t{
		data: cStrPath,
		size: C.size_t(len(path)),
	})

	if result.status != 0 {
		ch := *(**C.char)(unsafe.Pointer(&result.anon0[0]))
		defer C.free(unsafe.Pointer(ch))
		return Model{}, ModelOpenError(C.GoString(ch))
	}

	handle := *(*C.FastText_Handle_t)(unsafe.Pointer(&result.anon0[0]))

	return Model{
		p: handle,
	}, nil
}

// Closes a model handle
func (handle *Model) Close() error {
	C.FastText_DeleteHandle(handle.p)
	return nil
}

func (handle *Model) MultiLinePredict(lines []string, k int32, threshoad float32) ([]Predictions, error) {
	predics := make([]Predictions, 0, len(lines))

	for _, line := range lines {
		predictions, err := handle.Predict(line, k, threshoad)
		if err != nil && errors.Is(err, ErrPredictionFailed) {
			return nil, err
		}

		predics = append(predics, predictions)
	}

	if len(predics) == 0 {
		return nil, ErrNoPredictions
	}

	return predics, nil
}

// Perform model prediction
func (handle *Model) Predict(query string, k int32, threshoad float32) (Predictions, error) {
	var pinner runtime.Pinner
	defer pinner.Unpin()

	inputs := make([]C.FastText_PredictItem_t, k)
	inputsPtr := unsafe.SliceData(inputs)
	pinner.Pin(inputsPtr)

	count := C.FastText_Predict(
		handle.p,
		C.FastText_String_t{
			data: cStr(query),
			size: C.size_t(len(query)),
		},
		C.uint32_t(k),
		C.float(threshoad),
		inputsPtr,
	)

	if count == 0 {
		return nil, ErrNoPredictions
	}

	predictions := make(Predictions, k)

	for i := int32(0); i < k; i++ {
		str := C.GoStringN(inputs[i].lang.data, C.int(inputs[i].lang.size))

		predictions[i] = Prediction{
			Label:       str,
			Probability: float32(inputs[i].probability),
		}
	}

	return predictions, nil
}

func (handle Model) Wordvec(word string) []float32 {
	var pinner runtime.Pinner
	defer pinner.Unpin()

	strData := cStr(word)
	pinner.Pin(strData)

	r := C.FastText_Wordvec(
		handle.p,
		C.FastText_String_t{
			data: strData,
			size: C.size_t(len(word)),
		},
	)

	defer C.FastText_FreeFloatVector(r)

	vectors := make([]float32, r.size)
	ptr := (*float32)(unsafe.Pointer(r.data))
	copy(vectors, unsafe.Slice(ptr, r.size))
	return vectors
}

func (handle Model) Sentencevec(sentence string) []float32 {
	var pinner runtime.Pinner
	defer pinner.Unpin()

	strData := cStr(sentence)
	pinner.Pin(strData)

	r := C.FastText_Sentencevec(
		handle.p,
		C.FastText_String_t{
			data: strData,
			size: C.size_t(len(sentence)),
		},
	)

	defer C.FastText_FreeFloatVector(r)

	vectors := make([]float32, r.size)
	ptr := (*float32)(unsafe.Pointer(r.data))
	copy(vectors, unsafe.Slice(ptr, r.size))
	return vectors
}

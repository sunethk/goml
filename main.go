// Package goml contains sub-packages for various ML constructs in Go.
package main

import (
	"flag"
	"fmt"
	"github.com/kujenga/goml/lin"
	"github.com/kujenga/goml/mnist"
	"github.com/kujenga/goml/neural"
	"image"
	"image/png"
	"os"
	"time"
)

func main() {
	dataDir := flag.String("data-dir", "testdata/mnist", "Directory containing mnist training data as .gz files")
	imageFile := flag.String("image", "", "File name of 28 x 28 PNG file to evaluate")
	dataSize := flag.Int("data-size", 20000, "Training dataset size, maximum is 60,000")

	flag.Parse()

	m := neural.MLP{
		LearningRate: 0.1,
		Layers: []*neural.Layer{
			// Input
			{Name: "input", Width: 28 * 28},
			// Hidden
			{Name: "hidden1", Width: 100},
			// Output
			{Name: "output", Width: 10},
		},
		Introspect: func(s neural.Step) {
			//t.Logf("%+v", s)
			//fmt.Printf("%+v\n", s)
		},
	}

	if *imageFile == "" {
		fmt.Println("Please provide a file name of 28 x 28 PNG file to evaluate with flag -image")
		return
	}

	train(&m, dataDir, dataSize)

	frame := make(lin.Frame, 1)
	frame[0] = make([]float32, 28*28)

	fmt.Printf("Reading the image file %s\n", *imageFile)
	pixels := dataFromImage(*imageFile)
	for i, pixel := range pixels {
		frame[0][i] = float32(pixel)
	}

	fmt.Println("Predicting the digit...")
	predict := m.Predict(frame)
	fmt.Println("Predicting the digit... done!")
	//fmt.Printf("%+v\n", predict[0])

	fmt.Println("Probability vector")
	for i, probability := range predict[0] {
		fmt.Printf("%d -> %.6f\n", i, probability)
	}

	var maximum = float32(0)
	var digit = -1
	for i, i2 := range predict[0] {
		if i2 > maximum {
			maximum = i2
			digit = i
		}
	}
	if digit != -1 {
		fmt.Printf("\nDetected digit is %d\n", digit)
	} else {
		fmt.Printf("Cannot identify the digit!")
	}
	fmt.Println()
}

func train(m *neural.MLP, dataDir *string, dataSize *int) (*neural.MLP, bool) {
	fmt.Printf("Reading MNIST data from %s\n", *dataDir)
	dataset, err := mnist.Read(*dataDir)
	if err != nil {
		fmt.Printf("Error: %+v\n", err)
		return m, true
	}

	// Training and validation

	const epochs = 5

	fmt.Printf("Start training the network with dataset size of %d\n", *dataSize)
	t1 := time.Now()
	// NOTE: Dataset size is limited to speed up tests.
	_, err = m.Train(epochs, dataset.TrainInputs[:*dataSize], dataset.TrainLabels[:*dataSize])
	if err != nil {
		fmt.Printf("Error while training the network, %+v", err)
		return m, true
	}
	elapsed := time.Since(t1)
	fmt.Printf("Traing is complete...\nTime taken to train: %s\n", elapsed)
	return m, false
}

func dataFromImage(filePath string) (pixels []float64) {
	// read the file
	imgFile, err := os.Open(filePath)
	defer imgFile.Close()
	if err != nil {
		fmt.Println("Cannot read file:", err)
	}
	img, err := png.Decode(imgFile)
	if err != nil {
		fmt.Println("Cannot decode file:", err)
	}

	// create a grayscale image
	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			var rgba = img.At(x, y)
			//fmt.Printf("Color %+v\n", rgba)
			gray.Set(x, y, rgba)
		}
	}
	// make a pixel array
	pixels = make([]float64, len(gray.Pix))
	// populate the pixel array subtract Pix from 255 because
	// that's how the MNIST database was trained (in reverse)
	for i := 0; i < len(gray.Pix); i++ {
		pixels[i] = (float64(255-gray.Pix[i]) / 255.0 * 0.99) + 0.01
	}
	return
}

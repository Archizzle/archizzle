---
title: CS 180 Project 2
draft: false
tags:
    - CS 180
    - Computer Vision
    - Image Pyramid
---

## Part 1: Fun with Filters

### Part 1.1: Convolutions from scratch

The times for the runs were the following:
- 4 loops time: 2m 18.17s
- 2 loops time: 55.61s
- SciPy time: 1.06s

```py
def conv2d_4loops(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pt, pb = kh // 2, kh - 1 - kh // 2
    pl, pr = kw // 2, kw - 1 - kw // 2
    padded = np.pad(image, ((pt, pb), (pl, pr)), mode='constant', constant_values=0)
    output_h = h
    output_w = w
    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            conv_val = 0
            for m in range(kh):
                for n in range(kw):
                    conv_val += padded[i + m, j + n] * kernel[m, n]
            output[i, j] = conv_val

    return output
```
```py
def conv2d_2loops(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pt, pb = kh // 2, kh - 1 - kh // 2
    pl, pr = kw // 2, kw - 1 - kw // 2
    padded = np.pad(image, ((pt, pb), (pl, pr)), mode='constant', constant_values=0)
    output_h = h
    output_w = w
    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output
```

<table style="width:90%; table-layout:fixed;">
    <tr>
        <td colspan="3" style="text-align:center; width:33%;">
            <figure style="margin:0;">
                <figcaption>Original</figcaption>
                <img src="cs180_proj2_images/part1_2/original.png" alt="original" style="width:30%">
                <!-- Note: We're shrinking it by multiplying 33.3 by 0.9 -->
            </figure>
        </td>
    </tr>
    <!-- Alternate way to make centered -->
    <!-- <tr>
        <td style="text-align:center;">
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Original</figcaption>
                <img src="cs180_proj2_images/part1_1/original.png" alt="original">
            </figure>
        </td>
        <td style="text-align:center;">
        </td>
    </tr> -->
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>4 Loop Box Filter</figcaption>
                <img src="cs180_proj2_images/part1_1/box_4loops.png" alt="Box 4 loops">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>4 Loop Dx Filter</figcaption>
                <img src="cs180_proj2_images/part1_1/gx_4loops.png" alt="Dx 4 loops">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>4 Loop Dy Filter</figcaption>
                <img src="cs180_proj2_images/part1_1/gy_4loops.png" alt="Dy 4 loops">
            </figure>
        </td>
    </tr>
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>2 Loop Box Filter</figcaption>
                <img src="cs180_proj2_images/part1_1/box_2loops.png" alt="Box 2 loops">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>2 Loop Dx Filter</figcaption>
                <img src="cs180_proj2_images/part1_1/gx_2loops.png" alt="Dx 2 loops">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>2 Loop Dy Filter</figcaption>
                <img src="cs180_proj2_images/part1_1/gy_2loops.png" alt="Dy 2 loops">
            </figure>
        </td>
    </tr>
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>SciPy Loop Box Filter</figcaption>
                <img src="cs180_proj2_images/part1_1/box_scipy.png" alt="Box SciPy">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>SciPy Dx Filter</figcaption>
                <img src="cs180_proj2_images/part1_1/gx_scipy.png" alt="Dx SciPy">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>SciPy Dy Filter</figcaption>
                <img src="cs180_proj2_images/part1_1/gy_scipy.png" alt="Dy SciPy">
            </figure>
        </td>
    </tr>
</table>

### Part 1.2 Finite Difference Operator
The finite difference kernel was used in the x and y directions to find the partial derivatives with respect to x and y. Those results were then treated as a 2D vector, and its magnitude was produced by finding its L2 norm. We then In other words, if the image was denoted by $A$, its gradient/edges, $E$, is computed by
$$
    E_x = A * \begin{bmatrix} 1 & -1 \end{bmatrix} \\
    E_y = A * \begin{bmatrix} 1 \\ -1 \end{bmatrix} \\
    E = \sqrt{(E_x)^2 + (E_y)^2}
$$

Then we threshold this value to remove as much noise as possible without compromising the true edges.

<table style="width: 90%; table-layout:fixed;">
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Ex</figcaption>
                <img src="cs180_proj2_images/part1_2/ex.png" alt="E_x">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Ey</figcaption>
                <img src="cs180_proj2_images/part1_2/ey.png" alt="E_y">
            </figure>
        </td>
    </tr>
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Combined Image</figcaption>
                <img src="cs180_proj2_images/part1_2/e.png" alt="Combined Image">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Filtered Image</figcaption>
                <img src="cs180_proj2_images/part1_2/e_thresholded.png" alt="Thresholded Image">
            </figure>
        </td>
    </tr>
</table>

### Part 1.3: Derivative of Gaussian (DoG) Filter

Since the image is still somewhat noisy, smoothing/blurring the image prior to convolving with the finite difference kernel helps remove some of that noise. Essentially, given a gaussian kernel $G$, we have

$$
    E_x = A * G * \begin{bmatrix} 1 & -1 \end{bmatrix} \\
    E_y = A * G * \begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

Because convolution is associative, we can either:
1. Apply the gaussian filter to the image and then find the gradients of the blurred image
2. Compute the derivative of the gaussian (DoG) by convolving the finite difference operators with the gaussian filter. Then, we can directly convolve the DoG filter to the image.

The results are as follows:
<table style="width: 90%; table-layout:fixed;">
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Approach 1 Unfiltered</figcaption>
                <img src="cs180_proj2_images/part1_3/approach1_unfiltered.png" alt="Approach 1 Unfiltered">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Approach 1 Filtered</figcaption>
                <img src="cs180_proj2_images/part1_3/approach1_filtered.png" alt="Approach 1 Filtered">
            </figure>
        </td>
    </tr>
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Approach 2 Unfiltered</figcaption>
                <img src="cs180_proj2_images/part1_3/approach2_unfiltered.png" alt="Approach 2 Unfiltered">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Approach 2 Filtered</figcaption>
                <img src="cs180_proj2_images/part1_3/approach2_filtered.png" alt="Approach 2 Filtered">
            </figure>
        </td>
    </tr>
</table>

## Part 2: Fun with Frequencies!

### Part 2.1: Image "Sharpening"
The approach is to filter the high frequency values, and emphasize them by "adding" it to the original image. To find the high frequency values, we compute `details = image - blurred`, and thus our `result` becomes `image + alpha * details`, where `alpha` is a tunable constant. This can be computed by performing a single convolution. If we denote $I$ as the identity convolution (an odd kernel with all zeros except for the center element being $1$).

$$
    S = I * A + \alpha (I * A - G * A) \\
    = (I + \alpha (I - G)) * A \\
    = ((1 + \alpha) - \alpha G) * A
$$

#### Taj Mahal

<table style="width: 90%; table-layout:fixed;">
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Original</figcaption>
                <img src="cs180_proj2_images/part2_1/taj_mahal_alpha_0.png" alt="Original">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>alpha = 1</figcaption>
                <img src="cs180_proj2_images/part2_1/taj_mahal_alpha_1.png" alt="alpha = 1">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>alpha = 2</figcaption>
                <img src="cs180_proj2_images/part2_1/taj_mahal_alpha_2.png" alt="alpha = 2">
            </figure>
        </td>
    </tr>
</table>

#### Clouds
<table style="width: 90%; table-layout:fixed;">
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Original</figcaption>
                <img src="cs180_proj2_images/part2_1/clouds_alpha_0.png" alt="Original">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>alpha = 1</figcaption>
                <img src="cs180_proj2_images/part2_1/clouds_alpha_1.png" alt="alpha = 1">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>alpha = 2</figcaption>
                <img src="cs180_proj2_images/part2_1/clouds_alpha_2.png" alt="alpha = 2">
            </figure>
        </td>
    </tr>
</table>

Again, here we can see that the mountain's features are more visible.

#### Resharpening a Blurred Image (Pyramids)
<table style="width: 90%; table-layout:fixed;">
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Original</figcaption>
                <img src="cs180_proj2_images/part2_1/pyramids_original.png" alt="Original">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Blurred</figcaption>
                <img src="cs180_proj2_images/part2_1/pyramids_blurred.png" alt="blurred">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Resharpened</figcaption>
                <img src="cs180_proj2_images/part2_1/pyramids_resharpened.png" alt="resharpened">
            </figure>
        </td>
    </tr>
</table>

The reconstructed image has quite a few artifacts since a lot of the finer details are lost to the blurring process. However, most of the features aren't washed away, unlike the blurred image.

### Part 2.2: Hybrid Images

To achieve a blurred image, I overlayed a low frequency and high frequency image. The low frequency image was produced by applying a gaussian blur. For the high frequency image, I took the difference between the original image and a blurred image.

#### Is it a bird? Is it a plane?
I used a Gaussian blur with mean 6 and variance 12.

<table style="width: 90%; table-layout:fixed;">
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Bird</figcaption>
                <img src="cs180_proj2_images/part2_2/bird.png" alt="Bird">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Plane</figcaption>
                <img src="cs180_proj2_images/part2_2/plane.png" alt="Plane">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Combined</figcaption>
                <img src="cs180_proj2_images/part2_2/combined.png" alt="Combined">
            </figure>
        </td>
    </tr>
</table>

#### It's Superman!
Again, I used a Gaussian blur with mean 6 and variance 12.

<table style="width: 90%; table-layout:fixed;">
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Clark Kent</figcaption>
                <img src="cs180_proj2_images/part2_2/original_clark.png" alt="Clark Kent">
                <img src="cs180_proj2_images/part2_2/original_clark_fourier.png" alt="Clark Kent Fourier">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Superman</figcaption>
                <img src="cs180_proj2_images/part2_2/original_superman.png" alt="Superman">
                <img src="cs180_proj2_images/part2_2/original_superman_fourier.png" alt="Superman Fourier">
            </figure>
        </td>
    </tr>
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>High Pass Clark Kent</figcaption>
                <img src="cs180_proj2_images/part2_2/high_pass_clark.png" alt="High Pass Clark Kent">
                <img src="cs180_proj2_images/part2_2/high_pass_fourier.png" alt="High Pass Clark Kent Fourier">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Low Pass Superman</figcaption>
                <img src="cs180_proj2_images/part2_2/low_pass_superman.png" alt="Low Pass Superman">
                <img src="cs180_proj2_images/part2_2/low_pass_fourier.png" alt="Low Pass Superman Fourier">
            </figure>
        </td>
    </tr>
    <tr>
        <td colspan="2" style="text-align:center;" style="text-align:center; width:0.45%;">
            <figure style="margin:0;">
                <figcaption>High Pass Clark Kent</figcaption>
                <img src="cs180_proj2_images/part2_2/combined_superman_clark.png" alt="Combined" style="width:45%">
                <img src="cs180_proj2_images/part2_2/combined_fourier.png" alt="Combined Fourier" style="width:45%">
            </figure>
        </td>
    </tr>
</table>

#### Mario (Failed Attempt)

This set of images just didn't seem to work since the retro Mario was much larger than the modern Mario. As such, a lot of the details got washed away in the background.

<table style="width: 90%; table-layout:fixed;">
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Mario 2D</figcaption>
                <img src="cs180_proj2_images/part2_2/mario_2d.png" alt="2D Mario">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Mario 3D</figcaption>
                <img src="cs180_proj2_images/part2_2/mario_3d.png" alt="3D Mario">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Combined (Failed)</figcaption>
                <img src="cs180_proj2_images/part2_2/combined_mario.png" alt="Combined">
            </figure>
        </td>
    </tr>
</table>

### Part 2.3 & 2.4: Gaussian + Laplacian Stacks and Multiresolution Blending

Each level of the Gaussian stack is produced by blurring the previous level. The Laplacian stack is formed by taking differences of adjacent layers of the Gaussian stack. The last layer of the Gaussian stack is appended to the end of the Laplacian stack, so both end up with the same number of layers.

Pictured are layers 0, 2, and 4 of each layer of the Laplacian stack.

<table style="width: 90%; table-layout:fixed;">
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Apple Layer 0</figcaption>
                <img src="cs180_proj2_images/part2_3/apple_0.png" alt="Apple at layer 0">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Orange Layer 0</figcaption>
                <img src="cs180_proj2_images/part2_3/orange_0.png" alt="Orange at layer 0">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Combined Layer 0</figcaption>
                <img src="cs180_proj2_images/part2_3/combined_0.png" alt="Combined layer 0">
            </figure>
        </td>
    </tr>
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Apple Layer 2</figcaption>
                <img src="cs180_proj2_images/part2_3/apple_2.png" alt="Apple at layer 2">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Orange Layer 2</figcaption>
                <img src="cs180_proj2_images/part2_3/orange_2.png" alt="Orange at layer 2">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Combined Layer 2</figcaption>
                <img src="cs180_proj2_images/part2_3/combined_2.png" alt="Combined layer 2">
            </figure>
        </td>
    </tr>
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Apple Layer 4</figcaption>
                <img src="cs180_proj2_images/part2_3/apple_4.png" alt="Apple at layer 4">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Orange Layer 4</figcaption>
                <img src="cs180_proj2_images/part2_3/orange_4.png" alt="Orange at layer 4">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Combined Layer 4</figcaption>
                <img src="cs180_proj2_images/part2_3/combined_4.png" alt="Combined layer 4">
            </figure>
        </td>
    </tr>
    <tr>
        <td colspan="3" style="text-align:center; width:33%;">
            <figure style="margin:0;">
                <figcaption>Final Result</figcaption>
                <img src="cs180_proj2_images/part2_3/final_result.png" alt="original" style="width:50%">
                <!-- Note: We're shrinking it by multiplying 33.3 by 0.9 -->
            </figure>
        </td>
    </tr>
</table>

We can also use irregular masks.

<table style="width: 90%; table-layout:fixed;">
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Duck</figcaption>
                <img src="cs180_proj2_images/part2_4/duck.png" alt="Duck">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>City</figcaption>
                <img src="cs180_proj2_images/part2_4/city.png" alt="City">
            </figure>
        </td>
    </tr>
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Duck Mask</figcaption>
                <img src="cs180_proj2_images/part2_4/duck_mask.png" alt="Duck Mask">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>What are you doing here?</figcaption>
                <img src="cs180_proj2_images/part2_4/duck_in_city.png" alt="Duck in City">
            </figure>
        </td>
    </tr>
</table>

<table style="width: 90%; table-layout:fixed;">
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Cat</figcaption>
                <img src="cs180_proj2_images/part2_4/cat.png" alt="Cat">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Ice Cream Cone</figcaption>
                <img src="cs180_proj2_images/part2_4/cone.png" alt="Ice Cream Cone">
            </figure>
        </td>
    </tr>
    <tr>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Cat Mask</figcaption>
                <img src="cs180_proj2_images/part2_4/cat_mask.png" alt="Cat Mask">
            </figure>
        </td>
        <td style="text-align:center;">
            <figure style="margin:0;">
                <figcaption>Purrfection</figcaption>
                <img src="cs180_proj2_images/part2_4/cat_in_cone.png" alt="Cat in Cone">
            </figure>
        </td>
    </tr>
</table>
EC761 - Information Processing and Compression Course Project

> ECG Compression Using DNN and Other Compression Methods

Ram Janam Yadav1and Sameer S Durgoji2

> 1181EC136 , Dept. of ECE, National Institute of Technology Karnataka
> 2181EC240 , Dept. of ECE, National Institute of Technology Karnataka

November 30, 2021

**Abstract**

Analysis of ECG or electrocardiogram is of great significance in the
modern medicine for the diagnosis of various cardiovascular diseases.
ECG compression should be real time and should achieve a high
compression ratio. In this project, we have implemented ECG compression
and reconstruction using two methods: Using Discrete Wavelet Transform
and Using Deep Neural Networks. The details of the algorithms used are
explained in this report. The performance of the implemented algorithm
is measured using two parameters : Compression Ratio and Percentage RMS
Difference. In the end, the results of both the approaches are compared.

**Keywords**\
Electrocardiogram, Compression, DWT, DNN,compression ratio,
Autoencoders. Percentage RMS difference

**Introduction**

The electrocardiogram or ECG is a graphic representation of the heart's
electrical activity, which is formed as the cardiac cells depolarize and
repolarize. A typical ECG cycle is defined by the various features (P,
Q, R, S, and T) of the electrical wave.

> ![](cdef3a6a38d84902a0bbc481f6fe7668/media/image1.png){width="2.5194444444444444in"
> height="1.2541666666666667in"}
>
> Figure 1: Important features of ECG signal
>
> The P wave marks the activation of the atria, which are the chambers
> of the heart that receive blood from the body. Next in the ECG cycle
> comes the QRS complex. The heart beat cycle is measured as the time
> between the second of the three parts of the QRS complex, the large R
> peak. The QRS complex represents the activation of the left ventricle.
> During the QRS complex, which lasts about 80 msec, the atria prepare
> for the next beat, and the ventricles relax in the long T wave. It is
> these features of the ECG signal by which a cardiologist uses to
> analyze the health of the heart and note various disorders, such as
> atrial flutter, fibrillation, and bundle branch blocks .
>
> ECG or electrocardiogram is widely used in the modern medicine in the
> diagnosis of cardiovascular disease. However, during ECG analysis,
> data of very large size has to be recorded and stored. It also takes a
> long time to record ECG data. If not compressed, the large data will
> lead to an increase in the cost of storage. So, it is necessary to
> compress the data so that storage and analysis efficiency of
> electrocardiogram is increased.
>
> **Compression Methods**
>
> Compression techniques may be either lossless or lossy. In the case of
> lossless compression, the data is well preserved and the original
> signal can be perfectly reconstructed. But the compression achieved
> will be low. However, in the case of lossy compression, high
> compression can be achieved but some data will be lost. In the case of
> ECG data compression, a data compression ratio as high as possible
> should be achieved. However, it is also required to minimize losing
> electrocardiogram information or not to lose necessary information. To
> summarize, the ECG data compression should have these three
> characteristics: 1) real time compression 2) high compression rate

1

The ECG compression algorithms can be classified into three classes:

> 1\) Direct Data Processing : In direct data processing, data
> compression is done by extracting only the important information and
> eliminating the redundant information in ECG. The methods such as
> Evolutionary Computation, Turning Point Scan-Along Polygonal
> Approximation, and Differential-Pulse Coding Modulation (EC, TP, SAPA
> and DPCM) algorithms are used.
>
> 2\) Transform method: In this method, the data is represented in a
> different domain using various transforms. By this, the size of the
> signal is re duced significantly. The signal is reconstructed using
> the inverse transforms. Mathematical function, such as Kanade Lucas
> Tomasi, Dis crete Cosine Transform, Fast Fourier Trans form (KLT, DCT
> and FFT) algorithms are used for data compression.
>
> 3\) Neural Network approaches: In the neural network approaches, data
> compression is usually done by extracting the feature information
> implied in ECG through self-learning. The deep neural networks based
> compression technique is more popular because of its strong
> adaptability, anti- interference, parallel processing and good quality
> of configurable waveform.

**Methodology**

In this project, it has been decided to implement ECG compression using
the two methods as shown below:

> • Using Discrete Wavelet Transform• Using Deep Neural Networks

For the implementation, Physionet dataset has been used.

**ECG Compression Using DWT**

In this part, ECG compression was done using discrete wavelet transform
and variable length encoding.

This compression and reconstruction algorithm consists of the following
steps:

> 1\) Pre processing the raw waveform: In this step, detrending of the
> signal is performed to remove any linear trends present in the data.
>
> 2\) Apply DWT: Discrete Wavelet Transform of 5 level of decomposition
> is applied to the detrended signal. The biothogonal wavelet family of
> bior4.4 is used for the decomposition.
>
> 3\) Thresholding the coefficients and scaling them in the range of
> \[0,1\]: After decomposition, there will be many very small
> coefficients which are close to zero and just a few coefficients rep
> resent most of the total energy. So, coefficient thresholding is done
> to retain the given energy percentages.cA5 : 99.9%, cD5: 97%, cD1 to
> cD4: 85%. \[1\] A binary map is created where at the index of the
> zeroed coefficients, a 0 is assigned in the binary map; else a 1 will
> be as signed. Once thresholded, the coefficients are scaled to the
> range of \[0,1\] and the scaling fac tors are stored.
>
> 4\) Calculate minimum number of bits for quantization (N) : In this
> step, the problem statement would be to minimize the number of bits
> for quantization with a constraint of a maximum PRD defined at the
> begining.
>
> 5\) Quantization using N bits : Once the minimum number of bits for
> quantization (N) is determined, quantization of the coefficients is
> done with N bits.
>
> 6\) Combine and compress the coefficients : The coefficients and the
> binary map will be com bined into a single array. The quantized co
> efficients will be compressed using bit packing algorithm. The binary
> map will be compressed using variable run length encoding. This com
> pletes the process of encoding and the signal is ready for
> transmission. The compression ratio is calculated.
>
> 7\) Decompress and unscale the coefficients : This begins the process
> of reconstruction. The coefficients and the binary map is decompressed
> by repeating the compression step in reverse. Then the coefficients
> which are now in the range of \[0,1\] are unscaled to their original
> values using the scaling factors that were stored.
>
> 8\) Wavelet reconstruction : The signal is recon structed from the
> unscaled coefficients.
>
> **ECG Compression Using Deep Convolutional Autoencoders**
>
> In this Part We have proposed the method of ECG compression using Deep
> convolutional Autoencoders.
>
> Below are step involved in this method:
>
> 1\) Model contain the overall 21 layers and it has two parts one is
> encoder section which is used for compression and other is decoder
> section which is used for reconstruction of the ecg signals.

2

> ![](cdef3a6a38d84902a0bbc481f6fe7668/media/image2.png){width="1.8902777777777777in"
> height="2.0305544619422573in"}
>
> Figure 2: Model summary for apnea dataset
>
> 2\) The encoder section consists of total 11 layers which have input
> layer, convolutional layer and maxpooling layer. all the convolutional
> layers have relu activation function.
>
> 3\) The decoder section consists of total 10 layers which have
> convolutional layers, upsampling layers and finally the ouput layer
> which gives reconstructed signals as output.
>
> 4\) Hyperparameters used in the autoencoder model are (i)Adam
> optimizer (ii)mean square error as loss function. It was trained on
> 500 epochs.
>
> 5\) The dataset used for these methods is (i)Apnea ECG dataset
> (ii)MIT-BIH dataset.
>
> 6\) The autoencoder model provided a compres sion ratio of 32.

**Evaluation Criteria**

In this section we discuss about evaluation criteria used for evaluating
the performance of our implemented algorithm.

> 1\. Compression Ratio(CR): - Compression ratio is the ratio of
> original file and encoded file which gives us the compression of ecg
> signals.
>
> 2\. Percentage RMS difference(PRD): - It is percentage RMS difference
> between the original ecg signal and reconstructed ecg signal it should
> be as low as possible for better performance of our model.

**Results and Observation**

> 1\. DWT Method: - Using the DWT method we got better reconstruction
> but not very high compression and compression ratio was about 7 to 8
> on an average as observerd in the figures given below.
>
> ![](cdef3a6a38d84902a0bbc481f6fe7668/media/image3.png){width="2.5194433508311462in"
> height="0.9388877952755905in"}
>
> Figure 3: Results Obtained
>
> 2\. Deep Convolutional Autoencoder method: - Using Deep convolutional
> autoencoder method we got better reconstruction as well as we got high
> compression ratio of about 32. so we can say that deep learning method
> performed bet ter than DWT method which is not based on deep learning.
>
> **Test Results on ECG signals**

+-----+----------+----------+----------+----------+---------+-------+
| 1\. | >        |          |          |          |         |       |
|     |  Results |          |          |          |         |       |
|     | > using  |          |          |          |         |       |
|     | > DWT    |          |          |          |         |       |
|     | > Method |          |          |          |         |       |
|     | > on     |          |          |          |         |       |
|     | > Apnea  |          |          |          |         |       |
|     | >        |          |          |          |         |       |
|     | Dataset: |          |          |          |         |       |
+=====+==========+==========+==========+==========+=========+=======+
| 2\. | > \-     |          |          |          |         |       |
|     | > Refer  |          |          |          |         |       |
|     | > fgure  |          |          |          |         |       |
|     | > 4,     |          |          |          |         |       |
|     | 5,6,7,8. |          |          |          |         |       |
+-----+----------+----------+----------+----------+---------+-------+
|     | >        |          |          |          |         |       |
|     |  Results |          |          |          |         |       |
|     | > using  |          |          |          |         |       |
|     | > DWT    |          |          |          |         |       |
|     | > Method |          |          |          |         |       |
|     | > on     |          |          |          |         |       |
|     | >        |          |          |          |         |       |
|     |  MIT-BIH |          |          |          |         |       |
+-----+----------+----------+----------+----------+---------+-------+
| 3\. | >        |          |          |          |         |       |
|     | Dataset: |          |          |          |         |       |
|     | > -      |          |          |          |         |       |
|     | > Refer  |          |          |          |         |       |
|     | > fgure  |          |          |          |         |       |
|     | > 9,10,1 |          |          |          |         |       |
|     | 1,12,13. |          |          |          |         |       |
+-----+----------+----------+----------+----------+---------+-------+
|     | Results  | > using  | > Deep   | Auto     | Method  |       |
|     |          |          |          | encoders |         |       |
+-----+----------+----------+----------+----------+---------+-------+
| 4\. | on       | >        | Dataset: | \-       | > Refer | fgure |
|     |          |  MIT-BIH |          |          |         |       |
+-----+----------+----------+----------+----------+---------+-------+
|     | >        |          |          |          |         |       |
|     |  14,15,1 |          |          |          |         |       |
|     | 6,17,18. |          |          |          |         |       |
+-----+----------+----------+----------+----------+---------+-------+
|     | >        |          |          |          |         |       |
|     |  Results |          |          |          |         |       |
|     | > using  |          |          |          |         |       |
|     | > Deep   |          |          |          |         |       |
|     | > Auto   |          |          |          |         |       |
|     | encoders |          |          |          |         |       |
|     | > Method |          |          |          |         |       |
|     | > on     |          |          |          |         |       |
+-----+----------+----------+----------+----------+---------+-------+
|     | Apnea    |          |          |          |         |       |
|     | Dataset: |          |          |          |         |       |
|     | - Refer  |          |          |          |         |       |
|     | fgure    |          |          |          |         |       |
|     | 19,20,2  |          |          |          |         |       |
|     | 1,22,23. |          |          |          |         |       |
+-----+----------+----------+----------+----------+---------+-------+

> **Conclusions**
>
> As we know analysis of ECG is very important in diagnosing various
> cardiovascular diseases, it becomes very important to efficiently
> record it and store it. In this project, we have implemented two
> different compression algorithms to efficiently compress ECG data.In
> the end, the results were obtained for the Physionet database and the
> results were compared for both the algorithms. From the comparison, we
> were able to conclude that the method based on Deep Convolutional
> Autoencoders gave better compression ratio when compared to the method
> based on DWT.
>
> **Acknowledgements**
>
> We would like to thank our course Instructor Prof. Aparna ma'am for
> the guidance and review given during the midterm evaluation which
> helped us improve on this project.

3

> ![](cdef3a6a38d84902a0bbc481f6fe7668/media/image4.png){width="2.5194444444444444in"
> height="1.4166666666666667in"}

Figure 4: The original and detrended signal for apnea\
![](cdef3a6a38d84902a0bbc481f6fe7668/media/image5.png){width="2.5194444444444444in"
height="1.4166666666666667in"}

Figure 5: The coefficients before and after thresholding for apnea\
![](cdef3a6a38d84902a0bbc481f6fe7668/media/image6.png){width="2.5194444444444444in"
height="1.4180555555555556in"}

Figure 6: The thresholded coefficients after scaling for apnea\
![](cdef3a6a38d84902a0bbc481f6fe7668/media/image7.png){width="2.5194444444444444in"
height="1.4166666666666667in"}

Figure 7: Reconstructed signal and original signal for various bits for
quantization for apnea\
![](cdef3a6a38d84902a0bbc481f6fe7668/media/image8.png){width="2.5194444444444444in"
height="1.4180544619422573in"}

> Figure 8: The reconstructed signal for apnea
>
> ![](cdef3a6a38d84902a0bbc481f6fe7668/media/image9.png){width="2.5194433508311462in"
> height="1.4166666666666667in"}
>
> Figure 9: The original and detrended signal
>
> ![](cdef3a6a38d84902a0bbc481f6fe7668/media/image10.png){width="2.5194433508311462in"
> height="1.4166666666666667in"}
>
> Figure 10: The coefficients before and after thresholding\
> ![](cdef3a6a38d84902a0bbc481f6fe7668/media/image11.png){width="2.5194433508311462in"
> height="1.4166666666666667in"}
>
> Figure 11: The thresholded coefficients after scaling
>
> ![](cdef3a6a38d84902a0bbc481f6fe7668/media/image12.png){width="2.5194433508311462in"
> height="1.4166666666666667in"}
>
> Figure 12: Reconstructed signal and original signal for various bits
> for quantization\
> ![](cdef3a6a38d84902a0bbc481f6fe7668/media/image13.png){width="2.5194433508311462in"
> height="1.4180544619422573in"}
>
> Figure 13: The reconstructed signal

4

> ![](cdef3a6a38d84902a0bbc481f6fe7668/media/image14.png){width="2.5194444444444444in"
> height="4.648611111111111in"}

Figure 14: Model Architecture for MIT-BIH Dataset\
![](cdef3a6a38d84902a0bbc481f6fe7668/media/image15.png){width="2.5194444444444444in"
height="0.6083333333333333in"}

Figure 15: Model loss vs epochs for training and validation for MIT-BIH
dataset\
![](cdef3a6a38d84902a0bbc481f6fe7668/media/image16.png){width="2.5194444444444444in"
height="0.5749989063867017in"}

Figure 16: Original Test ECG signal for MIT-BIH dataset\
![](cdef3a6a38d84902a0bbc481f6fe7668/media/image17.png){width="2.5194444444444444in"
height="0.5222222222222223in"}

Figure 17: Encoded ECG signal using encoder section for MIT-BIH dataset\
![](cdef3a6a38d84902a0bbc481f6fe7668/media/image18.png){width="2.5194444444444444in"
height="0.4791666666666667in"}

Figure 18: The reconstructed signal for MIT-BIH dataset

> ![](cdef3a6a38d84902a0bbc481f6fe7668/media/image2.png){width="1.8888888888888888in"
> height="2.0305544619422573in"}
>
> Figure 19: Model Architecture for Apnea dataset
>
> ![](cdef3a6a38d84902a0bbc481f6fe7668/media/image19.png){width="1.8888888888888888in"
> height="1.2847222222222223in"}
>
> Figure 20: Model loss vs epochs for training and validation for Apnea
> dataset\
> ![](cdef3a6a38d84902a0bbc481f6fe7668/media/image20.png){width="1.8888888888888888in"
> height="1.5277777777777777in"}
>
> Figure 21: Original Test ECG signal for Apnea dataset\
> ![](cdef3a6a38d84902a0bbc481f6fe7668/media/image21.png){width="1.8888888888888888in"
> height="1.370832239720035in"}
>
> Figure 22: Encoded ECG signal using encoder section for Apnea dataset\
> ![](cdef3a6a38d84902a0bbc481f6fe7668/media/image22.png){width="1.8888888888888888in"
> height="1.4291655730533683in"}

+---+------------+--------------------------------------+
| 5 | Figure 23: | > The reconstructed signal for Apnea |
+===+============+======================================+
|   | > dataset  |                                      |
+---+------------+--------------------------------------+

**References**

> \[1 \] Rajoub, Bashar. (2002). An efficient coding\
> algorithm for the compression of ECG signals\
> using the wavelet transform. IEEE transac\
> tions on bio-medical engineering. 49. 355-62.\
> 10.1109/10.991163.
>
> \[2 \] Ozal Yildirim, Ru San Tan , July 2018, U\
> Rajendra Acharya, An Efficient Compression\
> of ECG Signals Using Deep Convolutional Au\
> toencoders, Cognitive Systems Research.

6

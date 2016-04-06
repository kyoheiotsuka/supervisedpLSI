# supervised probablistic semantic Indexing (spLSI)

Implementation of supervised probablistic semantic indexing (spLSI) using EM algorithm for inference is provided in this class. Binay classification task together with topic distribution inference can be conducted simultaneously.
Although spLSI is an original method and no detailed description can be found on the internet, the basic algorithm is based on that of [supervised topic model](https://www.cs.princeton.edu/~blei/papers/BleiMcAuliffe2007.pdf). The difference lays in whether prior distribution is assumed or not.

## Sample Code

Sample code is available for applying spLSI and extracting topics from synthetically generated data.
Figures below show the obtained results after observing 1000 synthetically generated data.

![sample](https://raw.github.com/kyoheiotsuka/supervisedpLSI/master/result/0.bmp)
![sample](https://raw.github.com/kyoheiotsuka/supervisedpLSI/master/result/1.bmp)
![sample](https://raw.github.com/kyoheiotsuka/supervisedpLSI/master/result/2.bmp)
![sample](https://raw.github.com/kyoheiotsuka/supervisedpLSI/master/result/3.bmp)
![sample](https://raw.github.com/kyoheiotsuka/supervisedpLSI/master/result/4.bmp)
![sample](https://raw.github.com/kyoheiotsuka/supervisedpLSI/master/result/5.bmp)
![sample](https://raw.github.com/kyoheiotsuka/supervisedpLSI/master/result/6.bmp)
![sample](https://raw.github.com/kyoheiotsuka/supervisedpLSI/master/result/7.bmp)

![sample](https://raw.github.com/kyoheiotsuka/supervisedpLSI/master/result/topicWord.jpg)

spLSI class provided supports not only extracting topics from training data but also inferring document-topic distribution and label of unseen data. 

## Licence
[MIT](https://github.com/kyoheiotsuka/supervisedpLSI/blob/master/LICENSE)

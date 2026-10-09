# Blind AI-paper evaluation

For each row, enter `pass`, `partial`, or `fail` in `review.json`. System identities are in
`blind-key.json`; leave that file closed until review is complete.

## R001 · qasper_test_02a5acb484bda77ef32a13f5d93d336472cf8cd4

**Question:** Use the known paper "Information Extraction with Character-level Neural Networks and Free Noisy Supervision" (doc_id: "qasper_1612_04118") to answer: what existing databases were used?

**Reference:** database containing historical time series data

**Answer:** The inspected passage does not name specific databases.

**Citations:** qasper_1612_04118

**Evidence:**

- `qasper_1612_04118:10800-11754`

> nted an architecture for information extraction from text using a combination of an existing parser and a deep neural network. The architecture can boost the precision of a high-recall information extraction system. To train the neural network, we use measures of consistency between extracted data and existing databases as a form of noisy supervision. The architecture resulted in substantial improvements over a mature and highly tuned constraint-based information extraction system for financial language text. While we used time series databases to derive measures of consistency for candidate extractions, our set-up can easily be applied to a variety of other information extraction tasks for which potentially noisy reference data is available.  We would like to thank my managers Alex Bozic, Tim Phelan and Joshwini Pereira for supporting this project, as well as David Rosenberg from the CTO's office for providing access to GPU infrastructure.

**Verdict:** pending

**Notes:**

## R002 · qasper_test_02a5acb484bda77ef32a13f5d93d336472cf8cd4

**Question:** Use the known paper "Information Extraction with Character-level Neural Networks and Free Noisy Supervision" (doc_id: "qasper_1612_04118") to answer: what existing databases were used?

**Reference:** database containing historical time series data

**Answer:** existing databases

**Citations:** qasper_1612_04118

**Evidence:**

- `qasper_1612_04118:10800-10845`

> nted an architecture for information extracti

**Verdict:** pending

**Notes:**

## R003 · qasper_test_04914917d01c9cd8718cd551dc253eb3827915d8

**Question:** Use the known paper "A system for the 2019 Sentiment, Emotion and Cognitive State Task of DARPAs LORELEI project" (doc_id: "qasper_1905_00472") to answer: Did the system perform well on low-resource languages?

**Reference:** Unanswerable

**Answer:** Yes, the inspected passage reports that the fielded systems obtained the best results in both English and Spanish language evaluations of the SEC pilot task.

**Citations:** qasper_1905_00472

**Evidence:**

- `qasper_1905_00472:600-2200`

> on how to best allocate resources for an effective disaster response. However, these efforts may be severely limited by the availability of resources for the local language. The ongoing DARPA project Low Resource Languages for Emergent Incidents (LORELEI) aims to further language processing technologies for low resource languages in the context of such a humanitarian crisis. In this work, we describe our submission for the 2019 Sentiment, Emotion and Cognitive state (SEC) pilot task of the LORELEI project. We describe a collection of sentiment analysis systems included in our submission along with the features extracted. Our fielded systems obtained the best results in both English and Spanish language evaluations of the SEC pilot task.  The growing adoption of online technologies has created new opportunities for emergency information propagation BIBREF0 . During crises, affected populations post information about what they are experiencing, what they are witnessing, and relate what they hear from other sources BIBREF1 . This information contributes to the creation and dissemination of situational awareness BIBREF2 , BIBREF3 , BIBREF4 , BIBREF0 , and crisis response agencies such as government departments or public health-care NGOs can make use of these channels to gain insight into the situation as it unfolds BIBREF2 , BIBREF5 . Additionally, these organizations might also post time-sensitive crisis management information to help with resource allocation and provide status reports BIBREF6 . While many of these organizations recognize the value of the information found onl

**Verdict:** pending

**Notes:**

## R004 · qasper_test_04914917d01c9cd8718cd551dc253eb3827915d8

**Question:** Use the known paper "A system for the 2019 Sentiment, Emotion and Cognitive State Task of DARPAs LORELEI project" (doc_id: "qasper_1905_00472") to answer: Did the system perform well on low-resource languages?

**Reference:** Unanswerable

**Answer:** Yes

**Citations:** qasper_1604_02201

**Evidence:**

- `qasper_1604_02201:0-200`

> Transfer Learning for Low-Resource Neural Machine Translation  The encoder-decoder framework for neural machine translation (NMT) has been shown effective in large data scenarios, but is much less eff

**Verdict:** pending

**Notes:**

## R005 · qasper_test_0b10cfa61595b21bf3ff13b4df0fe1c17bbbf4e9

**Question:** Use the known paper "Fine-tuning BERT for Joint Entity and Relation Extraction in Chinese Medical Text" (doc_id: "qasper_1908_07721") to answer: How do they perform the joint training?

**Reference:** They train a single model that integrates a BERT language model as a shared parameter layer on NER and RC tasks.

**Answer:** They perform joint training by integrating the BERT language model as a shared parameter layer through a dynamic range attention mechanism, which allows the model to focus on task-specific words and obtain context-dependent representations. This approach enables the model to jointly learn named entity recognition (NER) and relation classification (RC) tasks.

**Citations:** qasper_1908_07721

**Evidence:**

- `qasper_1908_07721:4000-4600`

> mechanism, we construct task-specific MASK matrix to control the attention range of the last $K$ layers in BERT language model, leading to the model focusing on the words of the task. This process helps obtain the corresponding task-specific context-dependent representations. In this way, the modified BERT language model can be used as the shared parameter layer in joint learning NER and RC task. We call the modified BERT language model shared task representation encoder (STR-encoder) in the following paper.  To sum up, the main contributions of our work are summarized as follows:  We propose 

**Verdict:** pending

**Notes:**

## R006 · qasper_test_0b10cfa61595b21bf3ff13b4df0fe1c17bbbf4e9

**Question:** Use the known paper "Fine-tuning BERT for Joint Entity and Relation Extraction in Chinese Medical Text" (doc_id: "qasper_1908_07721") to answer: How do they perform the joint training?

**Reference:** They train a single model that integrates a BERT language model as a shared parameter layer on NER and RC tasks.

**Answer:** They describe two joint learning approaches: joint tagging and parameter sharing. Joint tagging transforms NER and RC into sequence tagging with a novel scheme. Parameter sharing reuses the feature extraction layer between NER and RC.

**Citations:** qasper_1908_07721

**Evidence:**

- `qasper_1908_07721:8400-10000`

> l network BIBREF28. However, These methods cannot utilize the joint features between entity and relation, resulting in lower generalization performance when compared with joint learning methods.  Joint entity and relation extraction tasks solve NER and RC simultaneously. Compared with pipeline methods, joint learning methods are able to capture the joint features between entities and relations BIBREF29.  State-of-the-art joint learning methods can be divided into two categories, i.e., joint tagging and parameter sharing methods. Joint tagging transforms NER and RC tasks into sequence tagging tasks through a specially designed tagging scheme, e.g., novel tagging scheme proposed by Zheng et al. BIBREF3. Parameter sharing mechanism shares the feature extraction layer in the models of NER and RC. Compared to joint tagging methods, parameter sharing methods are able to effectively process multi-map problem. The most commonly shared parameter layer in medical domain is the Bi-LSTM network BIBREF9. However, compared with language model, the feature extraction ability of Bi-LSTM is relatively weaker, and the model cannot obtain pre-training knowledge through a large amount of unsupervised corpora, which further reduces the robustness of extracted features.  In this section, we introduce classic BERT language model and how to dynamically adjust the range of attention. On this basis, we propose a focused attention model for joint entity and relation extraction.  BERT is a language model that utilizes bidirectional attention mechanism and large-scale unsupervised corpora to obtain eff

**Verdict:** pending

**Notes:**

## R007 · qasper_test_0b5a505c1fca92258b9e83f53bb8cfeb81cb655a

**Question:** Use the known paper "On the Computational Power of RNNs" (doc_id: "qasper_1906_06349") to answer: What are edge weights?

**Reference:** Unanswerable

**Answer:** The inspected passage discusses edge weights in the context of infinite precision and hidden node activation functions, but does not define or describe edge weights themselves.

**Citations:** qasper_1906_06349

**Evidence:**

- `qasper_1906_06349:37000-38600`

> s to use the fact that INLINEFORM0 near INLINEFORM1 , and in fact this idea can be used for any activation function with a well-behaved Taylor series expansion around INLINEFORM2 .  Discussion 2.15. We “cheated" a little bit by allowing INLINEFORM0 edge weights and by having INLINEFORM1 where INLINEFORM2 wasn't quite linear. However, INLINEFORM3 edge weights make sense in the context of allowing infinite precision, and simple nonlinear functions over the hidden nodes are often used in practice, like the common softmax activation function.  We recognize two main avenues for further research. The first is to remove the necessity for infinite edge weights in the proof of Theorem 2.13, and the second is to extend the results of Theorems 1.11 and 2.13 to Turing recognizable languages.  In the proof of Lemma 2.11, edge weights of INLINEFORM0 are necessary for determining whether a hidden node ever becomes negative. Merely using large but finite weights does not suffice, because the values in the hidden state that they will be multiplied with are rapidly decreasing. Their product will vanish, and thus we would not be able to utilize the squashing properties of common activation functions as we did in the proof of Lemma 2.11. Currently we believe that it is possible to prove that GRUs are as computationally powerful as PDAs without using infinite edge weights, but are unaware of a method to do so.  Because to the our knowledge there is no analogue of the Chomsky-Sch INLINEFORM0 tzenberger Theorem for Turing recognizable languages, it seems difficult to directly extend our methods t

**Verdict:** pending

**Notes:**

## R008 · qasper_test_0b5a505c1fca92258b9e83f53bb8cfeb81cb655a

**Question:** Use the known paper "On the Computational Power of RNNs" (doc_id: "qasper_1906_06349") to answer: What are edge weights?

**Reference:** Unanswerable

**Answer:** Edge weights are defined as $w_{i,j} = score(i) + score(j)$ for each edge $(i, j) \in G$.

**Citations:** qasper_1712_00733

**Evidence:**

- `qasper_1712_00733:16600-16900`

> the edge direction and edge type (relation type), and define the importance of edge $w_{i,j}$ as the weights sum of two connected nodes as   $$w_{i,j}=score(i)+score(j), \quad \forall (i,j) \in G.$$   (Eq. 9)   In this paper, we take the top- $N$ edges ranked by $w_{i,j}$ as the final candidate know

**Verdict:** pending

**Notes:**

## R009 · qasper_test_0b5a7ccf09810ff5a86162d502697d16b3536249

**Question:** Use the known paper "SEPT: Improving Scientific Named Entity Recognition with Span Representation" (doc_id: "qasper_1911_03353") to answer: What simplification of the architecture is performed that resulted in same performance?

**Reference:** randomly sampling them rather than enumerate them all, simple max-pooling to extract span representation because those features are implicitly included in self-attention layers of transformers

**Answer:** The simplified architecture combines the span extractor with BERT.

**Citations:** qasper_1911_03353

**Evidence:**

- `qasper_1911_03353:400-600`

> odels, the performance of span extractors appears to become similar to sequence labeling models. To keep the advantages of span representation, we modified the model by under-sampling to balance the p

**Verdict:** pending

**Notes:**

## R010 · qasper_test_0b5a7ccf09810ff5a86162d502697d16b3536249

**Question:** Use the known paper "SEPT: Improving Scientific Named Entity Recognition with Span Representation" (doc_id: "qasper_1911_03353") to answer: What simplification of the architecture is performed that resulted in same performance?

**Reference:** randomly sampling them rather than enumerate them all, simple max-pooling to extract span representation because those features are implicitly included in self-attention layers of transformers

**Answer:** The simplification of the architecture involved removing the pruner, under-sampling to balance positive and negative samples, and using a simple pooling layer to extract span representation instead of more complex mechanisms like additional attention or LSTM networks.

**Citations:** qasper_1911_03353

**Evidence:**

- `qasper_1911_03353:3593-3633`

> simplify the origin network architecture

**Verdict:** pending

**Notes:**

## R011 · qasper_test_11360385dff0a9d7b8f4b106ba2b7fe15ca90d7c

**Question:** Use the known paper "THUEE system description for NIST 2019 SRE CTS Challenge" (doc_id: "qasper_1912_11585") to answer: What was the baseline?

**Reference:** Unanswerable

**Answer:** Unanswerable

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R012 · qasper_test_11360385dff0a9d7b8f4b106ba2b7fe15ca90d7c

**Question:** Use the known paper "THUEE system description for NIST 2019 SRE CTS Challenge" (doc_id: "qasper_1912_11585") to answer: What was the baseline?

**Reference:** Unanswerable

**Answer:** *No submitted answer*

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R013 · qasper_test_11c4071d9d7efeede84f47892b1fa0c6a93667eb

**Question:** Use the known paper "Neural Machine Translation with Imbalanced Classes" (doc_id: "qasper_2004_02334") to answer: What datasets do they look at?

**Reference:** Europarl v9, NewsTest2013 , NewsTest2014

**Answer:** They look at out-of-domain datasets.

**Citations:** qasper_2004_02334

**Evidence:**

- `qasper_2004_02334:7000-7200`

> stribution, model performance generally degrades. It is not surprising that frequency-biased classifiers show particular degradation in domain mismatch scenarios, as types that were infrequent in the 

**Verdict:** pending

**Notes:**

## R014 · qasper_test_11c4071d9d7efeede84f47892b1fa0c6a93667eb

**Question:** Use the known paper "Neural Machine Translation with Imbalanced Classes" (doc_id: "qasper_2004_02334") to answer: What datasets do they look at?

**Reference:** Europarl v9, NewsTest2013 , NewsTest2014

**Answer:** out-of-domain datasets

**Citations:** qasper_2004_02334

**Evidence:**

- `qasper_2004_02334:7000-8600`

> stribution, model performance generally degrades. It is not surprising that frequency-biased classifiers show particular degradation in domain mismatch scenarios, as types that were infrequent in the training distribution and were ignored by learning algorithm may appear with high frequency in the newer domain. koehn2017sixchallenges showed empirical evidence of poor generalization of NMT to out-of-domain datasets.  In other classification tasks, where each instance is classified independently, methods such as up-sampling the infrequent classes and down-sampling frequent classes are used. In NMT, since the classification is done within the context of sequences, it is possible to accomplish the objective of balancing by altering the lengths of sequences. This phenomenon of achieving balance by altering the sequence lengths is indirectly achieved by, e.g., BPE subword segmentation BIBREF5.  Quantification of Zipfian Imbalance: The class imbalance of an observed distribution of training classes is quantified as Divergence ($D$) from a balanced (uniform) distribution. Divergence is measured using a simplified version of Earth Mover Distance, in which the total cost for moving a probability mass between any two bins (analogous to class types) is the sum of the total mass moved. Since any mass moved out of one bin is moved into another, we divide the total per-bin mass moves in half to avoid double counting. Therefore, the imbalance measure $D$ on $K$ class distributions where $p_i$ is the observed probability of class $i$ in the training data is computed as:  The range of D is $

**Verdict:** pending

**Notes:**

## R015 · qasper_test_11dd2913d1517a1d47b367acb29fe9d79a9c95d1

**Question:** Use the known paper "Simple and Effective Noisy Channel Modeling for Neural Machine Translation" (doc_id: "qasper_1908_05731") to answer: How many parameters does their noisy channel model have?

**Reference:** Unanswerable

**Answer:** Unanswerable

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R016 · qasper_test_11dd2913d1517a1d47b367acb29fe9d79a9c95d1

**Question:** Use the known paper "Simple and Effective Noisy Channel Modeling for Neural Machine Translation" (doc_id: "qasper_1908_05731") to answer: How many parameters does their noisy channel model have?

**Reference:** Unanswerable

**Answer:** Unanswerable

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R017 · qasper_test_246e924017c48fa1f069361c44133fdf4f0386e1

**Question:** Use the known paper "Different Absorption from the Same Sharing: Sifted Multi-task Learning for Fake News Detection" (doc_id: "qasper_1909_01720") to answer: How is the selected sharing layer trained?

**Reference:** The selected sharing layer is trained jointly on the tasks of stance detection and fake news detection

**Answer:** The selected sharing layer is trained using a gated sharing cell to filter out useless features and an attention sharing cell to focus on features beneficial for specific tasks.

**Citations:** qasper_1909_01720

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R018 · qasper_test_246e924017c48fa1f069361c44133fdf4f0386e1

**Question:** Use the known paper "Different Absorption from the Same Sharing: Sifted Multi-task Learning for Fake News Detection" (doc_id: "qasper_1909_01720") to answer: How is the selected sharing layer trained?

**Reference:** The selected sharing layer is trained jointly on the tasks of stance detection and fake news detection

**Answer:** The selected sharing layer contains a gated sharing cell and an attention sharing cell. The gated sharing cell filters shared features from the shared layer, and the attention sharing cell focuses on features useful for specific tasks.

**Citations:** qasper_1909_01720

**Evidence:**

- `qasper_1909_01720:11600-13200`

> iate shared features for different tasks, we design a selected sharing layer following the shared layer. The selected sharing layer consists of two cells: gated sharing cell for filtering useless features and attention sharing cell for focusing on valuable shared features for specific tasks. The description of this layer is depicted in Figure FIGREF6 and Figure FIGREF15. In the following, we introduce two cells in details.  Gated Sharing Cell Inspired by forgotten gate mechanism of LSTM BIBREF32 and GRU BIBREF33, we design a single gated cell to filter useless shared features from shared layer. There are two reasons why we adopt single-gate mechanism. One is that transformer encoder in shared layer can efficiently capture the features of long-range dependencies. The features do not need to capture repeatedly by multiple complex gate mechanisms of LSTM and GRU. The other is that single-gate mechanism is more convenient for training BIBREF34. Formally, the gated sharing cell can be expressed as follows:  where ${\rm \textbf {H}}_{shared}\! \in \! \mathbb {R}^{1 \times l(d_p+d_w)}$ denotes the outputs of shared layer upstream, ${\rm \textbf {W}}_{fake} \in \mathbb {R}^{l(d_p+d_w) \times l(d_p+d_w)}$ and ${\rm \textbf {b}}_{fake} \in \mathbb {R}^{1 \times l(d_p+d_w)}$ are trainable parameters. $\sigma $ is a non-linear activation - sigmoid, which makes final choices for retaining and discarding features in shared layer.  Then the shared features after filtering via gated sharing cell ${\rm \textbf {g}}_{fake}$ for the task of fake news detection are represented as:  where $\odo

**Verdict:** pending

**Notes:**

## R019 · qasper_test_2d5d0b0c54105717bf48559b914fefd0c94964a6

**Question:** Use the known paper "Towards Supervised and Unsupervised Neural Machine Translation Baselines for Nigerian Pidgin" (doc_id: "qasper_2003_12660") to answer: What tokenization methods are used?

**Reference:** word-level , subword-level

**Answer:** The supervised models used both word-level and byte pair encoding (BPE) subword-level tokenization. The unsupervised model used only word-level tokenization.

**Citations:** qasper_2003_12660

**Evidence:**

- `qasper_2003_12660:2800-4400`

> seline models were trained using the Transformer architecture of BIBREF7. We experiment with both word-level and Byte Pair Encoding (BPE) subword-level tokenization methods for the supervised models. We learned 4000 byte pair encoding tokens, following the findings of BIBREF5. For the unuspervised model, we experiment with only word-level tokenization.  The dataset used for the supervised was obtained from the JW300 large-scale, parallel corpus for Machine Translation (MT) by BIBREF8. The train set contained 20214 sentence pairs, while the validation contained 1000 sentence pairs. Both the supervised and unsupervised models were evaluated on a test set of 2101 sentences preprocessed by the Masakhane group. The model with the highest test BLEU score is selected as the best.  Unsupervised model training followed BIBREF6 which used a Transformer of 4 encoder and 4 decoder layers with 10 attention heads. Embedding dimension was set to 300.  Supervised model training was performed with the open-source machine translation toolkit JoeyNMT by BIBREF9. For the byte pair encoding, embedding dimension was set to 256, while the embedding dimension was set to 300 for the word-level tokenization. The Transformer used for the byte pair encoding model had 6 encoder and 6 decoder layers, with 4 attention heads. For word-level, the encoder and decoder each had 4 layers with 10 attention heads for fair comparison to the unsupervised model. The models were each trained for 200 epochs on an Amazon EC2 p3.2xlarge instance.  English to Pidgin:  Pidgin to English:  For the word-level tokenization 

**Verdict:** pending

**Notes:**

## R020 · qasper_test_2d5d0b0c54105717bf48559b914fefd0c94964a6

**Question:** Use the known paper "Towards Supervised and Unsupervised Neural Machine Translation Baselines for Nigerian Pidgin" (doc_id: "qasper_2003_12660") to answer: What tokenization methods are used?

**Reference:** word-level , subword-level

**Answer:** The paper uses word-level and Byte Pair Encoding (BPE) subword-level tokenization methods.

**Citations:** qasper_2003_12660

**Evidence:**

- `qasper_2003_12660:2800-4400`

> seline models were trained using the Transformer architecture of BIBREF7. We experiment with both word-level and Byte Pair Encoding (BPE) subword-level tokenization methods for the supervised models. We learned 4000 byte pair encoding tokens, following the findings of BIBREF5. For the unuspervised model, we experiment with only word-level tokenization.  The dataset used for the supervised was obtained from the JW300 large-scale, parallel corpus for Machine Translation (MT) by BIBREF8. The train set contained 20214 sentence pairs, while the validation contained 1000 sentence pairs. Both the supervised and unsupervised models were evaluated on a test set of 2101 sentences preprocessed by the Masakhane group. The model with the highest test BLEU score is selected as the best.  Unsupervised model training followed BIBREF6 which used a Transformer of 4 encoder and 4 decoder layers with 10 attention heads. Embedding dimension was set to 300.  Supervised model training was performed with the open-source machine translation toolkit JoeyNMT by BIBREF9. For the byte pair encoding, embedding dimension was set to 256, while the embedding dimension was set to 300 for the word-level tokenization. The Transformer used for the byte pair encoding model had 6 encoder and 6 decoder layers, with 4 attention heads. For word-level, the encoder and decoder each had 4 layers with 10 attention heads for fair comparison to the unsupervised model. The models were each trained for 200 epochs on an Amazon EC2 p3.2xlarge instance.  English to Pidgin:  Pidgin to English:  For the word-level tokenization 

**Verdict:** pending

**Notes:**

## R021 · qasper_test_31894361833b3e329a1fb9ebf85a78841cff229f

**Question:** Use the known paper "Understanding the Radical Mind: Identifying Signals to Detect Extremist Content on Twitter" (doc_id: "qasper_1905_08067") to answer: Which textual features are used?

**Reference:** N-grams, word2vec

**Answer:** The inspected passage identifies three textual features: radical language, psychological signals, and behavioural features.

**Citations:** qasper_1905_08067

**Evidence:**

- `qasper_1905_08067:10600-12200`

> ated to radical activities. We identify three groups of features and then train a binary classifier to detect radical tweets.  Feature engineering is the process of exploring large spaces of heterogeneous features with the aim of discovering meaningful features that may aid in modeling the problem at hand. We explore three categories of information to identify relevant features to detect radical content. Some features are user-based while others are message-based. The three categories are: 1) Radical language (Textual features INLINEFORM0 ); 2) Psychological signals (Psychological features INLINEFORM1 ); and 3) Behavioural features ( INLINEFORM2 ). In the following, we detail each of these categories.  In order to understand how radical messages are constructed and used, as mentioned earlier, we analyze content of ISIS propaganda material published in Dabiq magazine. Dabiq is an online magazine published by ISIS terrorist groups with the purpose of recruiting people and promoting their propaganda and ideology. Using this data source, we investigate what topics, textual properties, and linguistic cues exist in these magazines. Our intuition is that utilising these linguistic cues from the extremist propaganda would allow us to detect supporters of ISIS group who are influenced by their propaganda.  We use two methods to extract the radical language from the propaganda corpus. First we calculate tf-idf scores for each gram in the propaganda corpus. We use uni-grams, bi-grams, and tri-grams to capture phrases and context in which words are being used. We then select the top sc

**Verdict:** pending

**Notes:**

## R022 · qasper_test_31894361833b3e329a1fb9ebf85a78841cff229f

**Question:** Use the known paper "Understanding the Radical Mind: Identifying Signals to Detect Extremist Content on Twitter" (doc_id: "qasper_1905_08067") to answer: Which textual features are used?

**Reference:** N-grams, word2vec

**Answer:** Textual features used include Radical language (Textual features INLINEFORM0), Psychological signals (Psychological features INLINEFORM1), and Behavioural features (INLINEFORM2).

**Citations:** qasper_1905_08067

**Evidence:**

- `qasper_1905_08067:10800-11300`

> eous features with the aim of discovering meaningful features that may aid in modeling the problem at hand. We explore three categories of information to identify relevant features to detect radical content. Some features are user-based while others are message-based. The three categories are: 1) Radical language (Textual features INLINEFORM0 ); 2) Psychological signals (Psychological features INLINEFORM1 ); and 3) Behavioural features ( INLINEFORM2 ). In the following, we detail each of these c

- `qasper_1905_08067:1000-1500`

> logical properties inferred from these material, and (3) we evaluate these models on Twitter to determine the extent to which it is possible to automatically identify online radical tweets. Our results show that radical users do exhibit distinguishable textual, psychological, and behavioural properties. We find that the psychological properties are among the most distinguishing features. Additionally, our results show that textual models using vector embedding features significantly improves the

- `qasper_1905_08067:11000-11500`

> ontent. Some features are user-based while others are message-based. The three categories are: 1) Radical language (Textual features INLINEFORM0 ); 2) Psychological signals (Psychological features INLINEFORM1 ); and 3) Behavioural features ( INLINEFORM2 ). In the following, we detail each of these categories.  In order to understand how radical messages are constructed and used, as mentioned earlier, we analyze content of ISIS propaganda material published in Dabiq magazine. Dabiq is an online m

**Verdict:** pending

**Notes:**

## R023 · qasper_test_3435e365adf7866e45670c865dc33bb7d2a6a0c6

**Question:** Use the known paper "Towards Understanding Gender Bias in Relation Extraction" (doc_id: "qasper_1911_03642") to answer: How are the sentences in WikiGenderBias curated?

**Reference:** Distant Supervision assumption, any sentence from an article written about one of those entities that mentions the other entity expresses the relation

**Answer:** The inspected passage does not describe how the sentences in WikiGenderBias are curated.

**Citations:** qasper_1911_03642

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R024 · qasper_test_3435e365adf7866e45670c865dc33bb7d2a6a0c6

**Question:** Use the known paper "Towards Understanding Gender Bias in Relation Extraction" (doc_id: "qasper_1911_03642") to answer: How are the sentences in WikiGenderBias curated?

**Reference:** Distant Supervision assumption, any sentence from an article written about one of those entities that mentions the other entity expresses the relation

**Answer:** The sentences in WikiGenderBias are specifically curated to analyze gender bias in relation extraction systems.

**Citations:** qasper_1911_03642

**Evidence:**

- `qasper_1911_03642:200-500`

> ile much attention has been dedicated towards improvements in accuracy, there have been no attempts in the literature to our knowledge to evaluate social biases in NRE systems. We create WikiGenderBias, a distantly supervised dataset with a human annotated test set. WikiGenderBias has sentences spec

**Verdict:** pending

**Notes:**

## R025 · qasper_test_34b434825f0ca3225dc8914f9da865d2b4674f08

**Question:** Use the known paper "A Context-Aware Approach for Detecting Check-Worthy Claims in Political Debates" (doc_id: "qasper_1912_08084") to answer: Does the baseline use any contextual information?

**Reference:** No

**Answer:** Yes, the baseline uses contextual features.

**Citations:** qasper_1912_08084

**Evidence:**

- `qasper_1912_08084:24000-25600`

> .  Table TABREF27 shows the results when using all features vs. excluding the contextual features vs. using the contextual features only. We can see that the contextual features have a major impact on performance: excluding them yields major drop for all measures, e.g., MAP drops from .427 to .385, and P@5 drops from .800 to .550. The last two rows in the table show that using contextual features only performs about the same as CB Platform (which uses no contextual features at all).  In the experiments above, we have been trying to predict whether a sentence is check-worthy in general, i.e., with respect to at least one source; this is how we trained and this is how we evaluated our models. Here, we want to evaluate how well our models perform at finding sentences that contain claims that would be judged as worthy for fact-checking with respect to each of the individual sources. The purpose is to see to what extent we can make our system potentially useful for a particular medium.  Another interesting question is whether we should use our generic system or we should retrain with respect to the target medium. Table TABREF31 shows the results for such a comparison, and it further compares to CB Platform. We can see that for all nine media, our model outperforms CB Platform in terms of MAP and P@50; this is also true for the other measures in most cases.  Moreover, we can see that training on all media is generally preferable to training on the target medium only, which shows that they do follow some common principles for selecting what is check-worthy; this means that a gener

**Verdict:** pending

**Notes:**

## R026 · qasper_test_34b434825f0ca3225dc8914f9da865d2b4674f08

**Question:** Use the known paper "A Context-Aware Approach for Detecting Check-Worthy Claims in Political Debates" (doc_id: "qasper_1912_08084") to answer: Does the baseline use any contextual information?

**Reference:** No

**Answer:** Yes

**Citations:** qasper_1912_08084

**Evidence:**

- `qasper_1912_08084:600-800`

>  ranking task. Unlike previous work, which has looked primarily at sentences in isolation, in this paper we focus on a rich input representation modeling the context: relationship between the target s

- `qasper_1912_08084:24000-24200`

> .  Table TABREF27 shows the results when using all features vs. excluding the contextual features vs. using the contextual features only. We can see that the contextual features have a major impact on

**Verdict:** pending

**Notes:**

## R027 · qasper_test_38363a7ed250bc729508c4c1dc975696a65c53cb

**Question:** Use the known paper "Can Neural Networks Learn Symbolic Rewriting?" (doc_id: "qasper_1911_04873") to answer: What translation models are explored?

**Reference:** NMT architecture BIBREF10

**Answer:** Neural machine translation (NMT) models are explored.

**Citations:** qasper_1911_04873

**Evidence:**

- `qasper_1911_04873:1200-2800`

> sticated toolchain developed for several decades BIBREF4 is surprisingly good BIBREF3, indicating that neural networks can learn quite complicated algorithms for symbolic data. This inspired us to pose a question: Can NMT models be used in the formal-to-formal setting? In particular: Can NMT models learn symbolic rewriting?  The answer is relevant to various tasks in automated reasoning. For example, neural models could compete with symbolic methods such as inductive logic programming BIBREF5 (ILP) that have been previously experimented with to learn simple rewrite tasks and theorem-proving heuristics from large formal corpora BIBREF6. Unlike (early) ILP, neural methods can however easily cope with large and rich datasets, without combinatorial explosion.  Our work is also an inquiry into the capabilities of NNs as such, in the spirit of works like BIBREF7.  To perform experiments answering our question we prepared two data sets – the first consists of examples extracted from proofs found by ATP (automated theorem prover) in a mathematical domain (AIM loops), whereas the second is a synthetic set of polynomial terms.  The data consists of sets of ground and nonground rewrites that came from Prover9 proofs of theorems about AIM loops produced by Veroff BIBREF8.  Many of the inferences in the proofs are paramodulations from an equation and have the form s = t  u[(s)] = vu[(t)] = v where $s, t, u, v$ are terms and $\theta $ is a substitution. For the most common equations $s = t$, we gathered corresponding pairs of terms $\big (u[\theta (s)], u[\theta (t)]\big )$ which were re

**Verdict:** pending

**Notes:**

## R028 · qasper_test_38363a7ed250bc729508c4c1dc975696a65c53cb

**Question:** Use the known paper "Can Neural Networks Learn Symbolic Rewriting?" (doc_id: "qasper_1911_04873") to answer: What translation models are explored?

**Reference:** NMT architecture BIBREF10

**Answer:** Neural Machine Translation (NMT) models are explored.

**Citations:** qasper_1911_04873

**Evidence:**

- `qasper_1911_04873:1000-1200`

> atical text in the goal is to translate it to its formal (computer understandable) counterpart. In particular, the NMT performance on a large synthetic -to-Mizar dataset produced by a relatively sophi

**Verdict:** pending

**Notes:**

## R029 · qasper_test_3ea82a5ca495ffbd1e30e8655aef1be4ba423efe

**Question:** Use the known paper "KPTimes: A Large-Scale Dataset for Keyphrase Generation on News Documents" (doc_id: "qasper_1911_12559") to answer: How do the editors' annotations differ from those in existing datasets?

**Reference:** Existing datasets are annotated by non-experts who use a larger, less controlled indexed vocabulary lacking the domain expertise shown by the editors

**Answer:** The inspected passage does not describe how the editors' annotations differ from those in existing datasets.

**Citations:** qasper_1911_12559

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R030 · qasper_test_3ea82a5ca495ffbd1e30e8655aef1be4ba423efe

**Question:** Use the known paper "KPTimes: A Large-Scale Dataset for Keyphrase Generation on News Documents" (doc_id: "qasper_1911_12559") to answer: How do the editors' annotations differ from those in existing datasets?

**Reference:** Existing datasets are annotated by non-experts who use a larger, less controlled indexed vocabulary lacking the domain expertise shown by the editors

**Answer:** The editors' annotations in KPTimes differ from those in existing datasets in that they are curated by editors rather than being non-expert annotations from authors. This difference is highlighted in the paper, which states that existing datasets are mainly from the scholarly domain and include non-expert annotations, while KPTimes uses editor-curated keyphrases.

**Citations:** qasper_1911_12559

**Evidence:**

- `qasper_1911_12559:200-1800`

> g datasets for keyphrase generation are only readily available for the scholarly domain and include non-expert annotations. In this paper we present KPTimes, a large-scale dataset of news texts paired with editor-curated keyphrases. Exploring the dataset, we show how editors tag documents , and how their annotations differ from those found in existing datasets. We also train and evaluate state-of-the-art neural keyphrase generation models on KPTimes to gain insights on how well they perform on the news domain. The dataset is available online at https:// github.com/ygorg/KPTimes.  Keyphrases are single or multi-word lexical units that best summarise a document BIBREF0. As such, they are of great importance for indexing, categorising and browsing digital libraries BIBREF1. Yet, very few documents have keyphrases assigned, thus raising the need for automatic keyphrase generation systems. This task falls under the task of automatic keyphrase extraction which can also be the subtask of finding keyphrases that only appear in the input document. Generating keyphrases can be seen as a particular instantiation of text summarization, where the goal is not to produce a well-formed piece of text, but a coherent set of phrases that convey the most salient information. Those phrases may or may not appear in the document, the latter requiring some form of abstraction to be generated. State-of-the-art systems for this task rely on recurrent neural networks BIBREF2, BIBREF3, BIBREF4, and hence require large amounts of annotated training data to achieve good performance. As gold annotated da

**Verdict:** pending

**Notes:**

## R031 · qasper_test_3f8a42eb0e904ce84c3fded2103f674e9cbc893d

**Question:** Use the known paper "Forget Me Not: Reducing Catastrophic Forgetting for Domain Adaptation in Reading Comprehension" (doc_id: "qasper_1911_00202") to answer: What is the training objective in the method introduced in this paper?

**Reference:** we explore strategies to reduce forgetting for comprehension systems during domain adaption. Our goal is to preserve the source domain's performance as much as possible, while keeping target domain's performance optimal and assuming no access to the source data. 

**Answer:** The paper does not describe the training objective of its proposed method.

**Citations:** qasper_1911_00202

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R032 · qasper_test_3f8a42eb0e904ce84c3fded2103f674e9cbc893d

**Question:** Use the known paper "Forget Me Not: Reducing Catastrophic Forgetting for Domain Adaptation in Reading Comprehension" (doc_id: "qasper_1911_00202") to answer: What is the training objective in the method introduced in this paper?

**Reference:** we explore strategies to reduce forgetting for comprehension systems during domain adaption. Our goal is to preserve the source domain's performance as much as possible, while keeping target domain's performance optimal and assuming no access to the source data. 

**Answer:** *No submitted answer*

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R033 · qasper_test_526dc757a686a1fe41e77f7e3848e3507940bfc4

**Question:** Use the known paper "Morfessor EM+Prune: Improved Subword Segmentation with Expectation Maximization and Pruning" (doc_id: "qasper_2003_03131") to answer: What is the improvement in performance brought by lexicon pruning on a simple EM algorithm?

**Reference:** Unanswerable

**Answer:** *No submitted answer*

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R034 · qasper_test_526dc757a686a1fe41e77f7e3848e3507940bfc4

**Question:** Use the known paper "Morfessor EM+Prune: Improved Subword Segmentation with Expectation Maximization and Pruning" (doc_id: "qasper_2003_03131") to answer: What is the improvement in performance brought by lexicon pruning on a simple EM algorithm?

**Reference:** Unanswerable

**Answer:** The paper describes an automatic procedure for tuning $\\alpha $ to reach a desired lexicon size, but it does not report performance improvements from pruning.

**Citations:** qasper_2003_03131

**Evidence:**

- `qasper_2003_03131:16600-18200`

> pruning criteria:  ($\alpha $-weighted) MDL pruning,  MDL with automatic tuning of $\alpha $ for lexicon size,  lexicon size with omitted prior or pretuned $\alpha $.  In ($\alpha $-weighted) Minimum Description Length (MDL) pruning, subwords are pruned until the estimated cost starts rising, or until the pruning quota for the iteration is reached, whichever comes first.  A subword lexicon of a predetermined size can be used as pruning criterion in two different ways. If the desired $\alpha $ is known in advance, or if the prior is omitted, subwords are pruned until the desired lexicon size is reached, or until the pruning quota for the iteration is reached, whichever comes first.  To reach a subword lexicon of a predetermined size while using the Morfessor prior, the new automatic tuning procedure can be applied. For each subword, the estimated change in prior and likelihood are computed separately. These allow computing the value of $\alpha $ that would cause the removal of each subword to be cost neutral, i.e. the value that would cause MDL pruning to terminate at that subword. For subwords with the same sign for both the change in prior and likelihood, no such threshold $\alpha $ can be computed: if the removal decreases both costs the subword will always be removed, and if it increases both costs it will always be kept. Sorting the list of subwords according to the estimated threshold $\alpha $ including the always kept subwords allows automatically tuning $\alpha $ so that a subword lexicon of exactly the desired size is retained after MDL pruning. The automatic tunin

**Verdict:** pending

**Notes:**

## R035 · qasper_test_538430077b1820011c609c8ae147389b960932c8

**Question:** Use the known paper "Adversarial Training for Aspect-Based Sentiment Analysis with BERT" (doc_id: "qasper_2001_11316") to answer: What aspects are considered?

**Reference:** Unanswerable

**Answer:** The inspected passage defines aspects as opinion targets, such as "spaghetti" in the example. It does not name specific aspects considered in the experiments.

**Citations:** qasper_2001_11316

**Evidence:**

- `qasper_2001_11316:1200-2800`

>  of adversarial training in ABSA.  Understanding what people are talking about and how they feel about it is valuable especially for industries which need to know the customers' opinions on their products. Aspect-Based Sentiment Analysis (ABSA) is a branch of sentiment analysis which deals with extracting the opinion targets (aspects) as well as the sentiment expressed towards them. For instance, in the sentence The spaghetti was out of this world., a positive sentiment is mentioned towards the target which is spaghetti. Performing these tasks requires a deep understanding of the language. Traditional machine learning methods such as SVM BIBREF2, Naive Bayes BIBREF3, Decision Trees BIBREF4, Maximum Entropy BIBREF5 have long been practiced to acquire such knowledge. However, in recent years due to the abundance of available data and computational power, deep learning methods such as CNNs BIBREF6, BIBREF7, BIBREF8, RNNs BIBREF9, BIBREF10, BIBREF11, and the Transformer BIBREF12 have outperformed the traditional machine learning techniques in various tasks of sentiment analysis. Bidirectional Encoder Representations from Transformers (BERT) BIBREF13 is a deep and powerful language model which uses the encoder of the Transformer in a self-supervised manner to learn the language model. It has been shown to result in state-of-the-art performances on the GLUE benchmark BIBREF14 including text classification. BIBREF1 show that adding domain-specific information to this model can enhance its performance in ABSA. Using their post-trained BERT (BERT-PT), we add adversarial examples to 

**Verdict:** pending

**Notes:**

## R036 · qasper_test_538430077b1820011c609c8ae147389b960932c8

**Question:** Use the known paper "Adversarial Training for Aspect-Based Sentiment Analysis with BERT" (doc_id: "qasper_2001_11316") to answer: What aspects are considered?

**Reference:** Unanswerable

**Answer:** *No submitted answer*

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R037 · qasper_test_58df55002fbcba76b9aeb2181d78378b8c01a827

**Question:** Use the known paper "Scalable Neural Dialogue State Tracking" (doc_id: "qasper_1910_09942") to answer: Which part of their architecture provides the most speedup in comparison to existing approaches?

**Reference:** Unanswerable

**Answer:** *No submitted answer*

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R038 · qasper_test_58df55002fbcba76b9aeb2181d78378b8c01a827

**Question:** Use the known paper "Scalable Neural Dialogue State Tracking" (doc_id: "qasper_1910_09942") to answer: Which part of their architecture provides the most speedup in comparison to existing approaches?

**Reference:** Unanswerable

**Answer:** The proposed G-SAT model reduces latency by over 15 times compared to state-of-the-art DST systems.

**Citations:** qasper_1910_09942

**Evidence:**

- `qasper_1910_09942:3400-5000`

>  Encoder - BIBREF9), simplifies the GLAD neural architecture removing the slot-specific recurrent and self-attention layers of the encoder, but still requires separate encoders for the utterance, the system action and the slot-values.  Although the neural network models mentioned above achieve state-of-the-art performance, the complexity of their architectures make them highly inefficient in terms of time complexity, with a significant latency in their prediction time. Such latency may soon become a serious limitation for their deployment into concrete application scenarios with increasing number of slots, where real time is a strong requirement. Along this perspective, this work investigates the time complexity of state-of-the-art DST models and addresses their current limitations. Our contributions are the following:  we have designed and implemented an efficient DST, consisting of a Global encoder and Slot-Attentive decoders (G-SAT);  we provide empirical evidences (three languages of the WOZ2.0 dataset BIBREF6) that the proposed G-SAT model considerably reduces the latency time with respect to state-of-art DST systems (i.e. over 15 times faster), while keeping the dialogue state prediction inline with such systems;  further experiments show that the proposed model is highly robust when either pre-trained embeddings are used or when they are not used, in this case outperforming state-of-art systems.  The implementation of the proposed G-SAT model is publicly available.  The paper is structured as follows. Section 2 summarizes the main concepts behind the definition of di

**Verdict:** pending

**Notes:**

## R039 · qasper_test_6073be8b88f0378cd0c4ffcad87e1327bc98b991

**Question:** Use the known paper "On Layer Normalization in the Transformer Architecture" (doc_id: "qasper_2002_04745") to answer: What is mean field theory?

**Reference:** Unanswerable

**Answer:** *No submitted answer*

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R040 · qasper_test_6073be8b88f0378cd0c4ffcad87e1327bc98b991

**Question:** Use the known paper "On Layer Normalization in the Transformer Architecture" (doc_id: "qasper_2002_04745") to answer: What is mean field theory?

**Reference:** Unanswerable

**Answer:** Mean field theory is used to analyze the initialization behavior of the Post-LN Transformer, showing that large gradients near the output layer make training unstable without a learning rate warm-up.

**Citations:** qasper_2002_04745

**Evidence:**

- `qasper_2002_04745:200-1800`

> ning rate warm-up stage, which is shown to be crucial to the final performance but will slow down the optimization and bring more hyper-parameter tunings. In this paper, we first study theoretically why the learning rate warm-up stage is essential and show that the location of layer normalization matters. Specifically, we prove with mean field theory that at initialization, for the original-designed Post-LN Transformer, which places the layer normalization between the residual blocks, the expected gradients of the parameters near the output layer are large. Therefore, using a large learning rate on those gradients makes the training unstable. The warm-up stage is practically helpful for avoiding this problem. On the other hand, our theory also shows that if the layer normalization is put inside the residual blocks (recently proposed as Pre-LN Transformer), the gradients are well-behaved at initialization. This motivates us to remove the warm-up stage for the training of Pre-LN Transformers. We show in our experiments that Pre-LN Transformers without the warm-up stage can reach comparable results with baselines while requiring significantly less training time and hyper-parameter tuning on a wide range of applications.  The Transformer BIBREF0 is one of the most commonly used neural network architectures in natural language processing. Layer normalization BIBREF1 plays a key role in Transformer's success. The originally designed Transformer places the layer normalization between the residual blocks, which is usually referred to as the Transformer with Post-Layer Normalization

**Verdict:** pending

**Notes:**

## R041 · qasper_test_7ae95716977d39d96e871e552c35ca0753115229

**Question:** Use the known paper "Represent, Aggregate, and Constrain: A Novel Architecture for Machine Reading from Noisy Sources" (doc_id: "qasper_1610_09722") to answer: what dataset did they use?

**Reference:** Event dataset with news articles

**Answer:** The Stanford Plane Crash Dataset BIBREF15 was used.

**Citations:** qasper_1610_09722

**Evidence:**

- `qasper_1610_09722:15600-16000`

> e passing the initial outgoing message for a variable INLINEFORM1 to its neighboring Exactly-1 factors is: DISPLAYFORM0   where the sigmoid moves the scores into probability space.  A message from an Exactly-1 factor to its neighboring variables is calculated as:   INLINEFORM0   All subsequent LBP iterations compute variable messages as in Eq. EQREF24 , incorporating the out-going factor beliefs o

**Verdict:** pending

**Notes:**

## R042 · qasper_test_7ae95716977d39d96e871e552c35ca0753115229

**Question:** Use the known paper "Represent, Aggregate, and Constrain: A Novel Architecture for Machine Reading from Noisy Sources" (doc_id: "qasper_1610_09722") to answer: what dataset did they use?

**Reference:** Event dataset with news articles

**Answer:** Stanford Plane Crash Dataset

**Citations:** qasper_1610_09722

**Evidence:**

- `qasper_1610_09722:15600-16000`

> e passing the initial outgoing message for a variable INLINEFORM1 to its neighboring Exactly-1 factors is: DISPLAYFORM0   where the sigmoid moves the scores into probability space.  A message from an Exactly-1 factor to its neighboring variables is calculated as:   INLINEFORM0   All subsequent LBP iterations compute variable messages as in Eq. EQREF24 , incorporating the out-going factor beliefs o

**Verdict:** pending

**Notes:**

## R043 · qasper_test_7ef7a5867060f91eac8ad857c186e51b767c734b

**Question:** Use the known paper "Hard but Robust, Easy but Sensitive: How Encoder and Decoder Perform in Neural Machine Translation" (doc_id: "qasper_1908_06259") to answer: How do they generate input noise for the encoder and decoder?

**Reference:** random dropping, random noising, random swapping

**Answer:** They simulate input noise with three operations: random dropping, random noising, and random swapping.

**Citations:** qasper_1908_06259

**Evidence:**

- `qasper_1908_06259:7600-9200`

>   In this section, we compare the characteristics between the encoder and decoder by analyzing their robustness according to the input noise in the inference phase. We simulate the input noise with three typical operations BIBREF20, BIBREF21: 1) random dropping: we randomly drop the input tokens of encoder and decoder respectively with different drop rates; 2) random noising: we randomly select tokens and replace its embedding with random noise; 3) random swapping: we randomly reverse the order for the adjacent tokens. The decoder in NMT model typically generates the current token one-by-one conditioned on the previous generated tokens, which suffers from error propagation BIBREF22: if a token is incorrectly predicted by the decoder, it will affect the prediction of the following tokens. Adding input noise to the decoder will further enhance the effect of error propagation, and thus influence our analysis. To eliminate the influence of error propagation, we apply teacher forcing BIBREF23 in the inference phase by feeding the previous ground-truth target tokens instead of the previously generated target tokens, following BIBREF24. We evaluate our model on IWSLT14 De$\leftrightarrow $En, IWSLT14 Ro$\leftrightarrow $En and WMT17 Chinese$\leftrightarrow $English (Zh$\leftrightarrow $En) translation tasks. More details on experimental configurations are described in supplementary materials (Section 1.2). The results are demonstrated in Figure FIGREF10. It can be seen that as the perturbation rate increases, adding different types of noise to the decoder input consistently achiev

**Verdict:** pending

**Notes:**

## R044 · qasper_test_7ef7a5867060f91eac8ad857c186e51b767c734b

**Question:** Use the known paper "Hard but Robust, Easy but Sensitive: How Encoder and Decoder Perform in Neural Machine Translation" (doc_id: "qasper_1908_06259") to answer: How do they generate input noise for the encoder and decoder?

**Reference:** random dropping, random noising, random swapping

**Answer:** They generate input noise for the encoder and decoder by randomly adding different levels of noise to the input of the encoder and decoder respectively during inference.

**Citations:** qasper_1908_06259

**Evidence:**

- `qasper_1908_06259:2400-2600`

>  than the encoder. These two results suggest that the decoder handles an easier task than the encoder in NMT.  The decoder is more sensitive to the input noise than the encoder. We randomly add differ

**Verdict:** pending

**Notes:**

## R045 · qasper_test_87523fb927354ddc8ad1357a81f766b7ea95f53c

**Question:** Use the known paper "Clotho: An Audio Captioning Dataset" (doc_id: "qasper_1910_09387") to answer: How many annotators did they have?

**Reference:** Unanswerable

**Answer:** 5

**Citations:** qasper_1910_09387

**Evidence:**

- `qasper_1910_09387:9000-9200`

> {z}=\lbrace c_{\text{sam}}^{z,u}\rbrace _{u=1}^{N_{\text{cp}}}$ for each $\mathbf {x}_{\text{sam}}^{z}$, where $c_{\text{sam}}^{z,u}$ is an eight to 20 words long caption for $\mathbf {x}_{\text{sam}}

**Verdict:** pending

**Notes:**

## R046 · qasper_test_87523fb927354ddc8ad1357a81f766b7ea95f53c

**Question:** Use the known paper "Clotho: An Audio Captioning Dataset" (doc_id: "qasper_1910_09387") to answer: How many annotators did they have?

**Reference:** Unanswerable

**Answer:** Unanswerable

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R047 · qasper_test_88757bc49ccab76e587fba7521f0981d6a1af2f7

**Question:** Use the known paper "Predicting Audience's Laughter Using Convolutional Neural Network" (doc_id: "qasper_1702_02584") to answer: What lexical cues are used for humor recogition?

**Reference:** Incongruity, Ambiguity, Interpersonal Effect, Phonetic Style

**Answer:** Lexical cues used for humor recognition include Incongruity, Ambiguity, Interpersonal Effect, and Phonetic Style.

**Citations:** qasper_1702_02584

**Evidence:**

- `qasper_1702_02584:2800-3300`

> d latent semantic structure features based on the following four aspects: (a) Incongruity, (b) Ambiguity, (c) Interpersonal Effect, and (d) Phonetic Style. In addition, Word2Vec BIBREF4 distributed representations were utilized in the model building.  Beyond lexical cues from text inputs, other research has also utilized speakers' acoustic cues BIBREF2 , BIBREF5 . These studies have typically used audio tracks from TV shows and their corresponding captions in order to categorize characters' spea

**Verdict:** pending

**Notes:**

## R048 · qasper_test_88757bc49ccab76e587fba7521f0981d6a1af2f7

**Question:** Use the known paper "Predicting Audience's Laughter Using Convolutional Neural Network" (doc_id: "qasper_1702_02584") to answer: What lexical cues are used for humor recogition?

**Reference:** Incongruity, Ambiguity, Interpersonal Effect, Phonetic Style

**Answer:** The inspected passage does not describe the lexical cues used for humor recognition.

**Citations:** qasper_1702_02584

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R049 · qasper_test_8cf5abf0126f19253930478b02f0839af28e4093

**Question:** Use the known paper "NRC-Canada at SMM4H Shared Task: Classifying Tweets Mentioning Adverse Drug Reactions and Medication Intake" (doc_id: "qasper_1805_04558") to answer: what are the sentiment features used?

**Reference:** the number of tokens with INLINEFORM0, the total score = INLINEFORM0, the maximal score = INLINEFORM0, the score of the last token in the tweet

**Answer:** The sentiment features used are the Hu and Liu Lexicon, Norms of Valence, Arousal, and Dominance, labMT, and NRC Emoticon Lexicon.

**Citations:** qasper_1805_04558

**Evidence:**

- `qasper_1805_04558:14400-16000`

> noun Lexicon features: the number of tokens from the Pronoun lexicon matched in the tweet;  domain word embeddings: the sum of the domain word embeddings for all tokens in the tweet;  domain word clusters: presence of tokens from the domain word clusters.  Sentiment Lexicon Features  We generated features using the sentiment scores provided in the following lexicons: Hu and Liu Lexicon BIBREF17 , Norms of Valence, Arousal, and Dominance BIBREF18 , labMT BIBREF19 , and NRC Emoticon Lexicon BIBREF20 . The first three lexicons were created through manual annotation while the last one, NRC Emoticon Lexicon, was generated automatically from a large collection of tweets with emoticons. The following set of features were calculated separately for each tweet and each lexicon:  the number of tokens with INLINEFORM0 ;  the total score = INLINEFORM0 ;  the maximal score = INLINEFORM0 ;  the score of the last token in the tweet.   We experimented with a number of other existing manually created or automatically generated sentiment and emotion lexicons, such as the NRC Emotion Lexicon BIBREF21 and the NRC Hashtag Emotion Lexicon BIBREF22 (http://saifmohammad.com/ WebPages/lexicons.html), but did not observe any improvement in the cross-validation experiments. None of the sentiment lexicon features were effective in the cross-validation experiments on Task 1; therefore, we did not include them in the final feature set for this task.  For each task, our team submitted three sets of predictions. The submissions differed in the sets of features and parameters used to train the classificatio

**Verdict:** pending

**Notes:**

## R050 · qasper_test_8cf5abf0126f19253930478b02f0839af28e4093

**Question:** Use the known paper "NRC-Canada at SMM4H Shared Task: Classifying Tweets Mentioning Adverse Drug Reactions and Medication Intake" (doc_id: "qasper_1805_04558") to answer: what are the sentiment features used?

**Reference:** the number of tokens with INLINEFORM0, the total score = INLINEFORM0, the maximal score = INLINEFORM0, the score of the last token in the tweet

**Answer:** The inspected passage describes general textual features and sentiment/domain-specific features but does not list the sentiment features used.

**Citations:** qasper_1805_04558

**Evidence:**

- `qasper_1805_04558:11000-12600`

> d. URLs and user mentions were normalized to http://someurl and @username, respectively. Tweets were tokenized with the CMU Twitter NLP tool. BIBREF11   The classification model leverages a variety of general textual features as well as sentiment and domain-specific features described below. Many features were inspired by previous work on ADR BIBREF12 , BIBREF8 , BIBREF9 and our work on sentiment analysis (such as the winning system in the SemEval-2013 task on sentiment analysis in Twitter BIBREF13 and best performing stance detection system BIBREF14 ).   General Textual Features  The following surface-form features were used:   INLINEFORM0 -grams: word INLINEFORM1 -grams (contiguous sequences of INLINEFORM2 tokens), non-contiguous word INLINEFORM3 -grams ( INLINEFORM4 -grams with one token replaced by *), character INLINEFORM5 -grams (contiguous sequences of INLINEFORM6 characters), unigram stems obtained with the Porter stemming algorithm;  General-domain word embeddings:  dense word representations generated with word2vec on ten million English-language tweets, summed over all tokens in the tweet,  word embeddings distributed as part of ConceptNet 5.5 BIBREF15 , summed over all tokens in the tweet;  General-domain word clusters: presence of tokens from the word clusters generated with the Brown clustering algorithm on 56 million English-language tweets; BIBREF11   Negation: presence of simple negators (e.g., not, never); negation also affects the INLINEFORM0 -gram features—a term INLINEFORM1 becomes INLINEFORM2 if it occurs after a negator and before a punctuation mark; 

**Verdict:** pending

**Notes:**

## R051 · qasper_test_95af7aaea3ce9dab4cf64e2229ce9b98381dd050

**Question:** Use the known paper "From narrative descriptions to MedDRA: automagically encoding adverse drug reactions" (doc_id: "qasper_1612_03762") to answer: Did they test the idea that the system reduces the time needed to encode ADR reports on real pharmacologists?

**Reference:** Yes

**Answer:** No

**Citations:** qasper_1612_03762

**Evidence:**

- `qasper_1612_03762:50600-52200`

> iness intelligence suite Pentaho. VigiSegn offers a large number of encoded ADRs. The encoding has been manually performed and validated by experts working at pharmacovigilance centres. Encoding results have then been sent to the national regulatory authority, AIFA.  We performed a test composed by the following steps.  We launch an ETL procedure through Pentaho Data Integration. Reports are transferred from VigiSegn to an ad hoc database TestDB. The dataset covers all the 4445 reports received, revised and validated during the year 2014 for the Italian region Veneto.  The ETL procedure extracts the narrative descriptions from reports stored in TestDB. For each description, the procedure calls MagiCoder from  VigiFarmaco; the output, i.e., a list of MedDRA terms, is stored in a table of TestDB.  Manual and automatic encodings of each report are finally compared through an SQL query. In order to have two uniform data sets, we compared only those reports where MagiCoder recognized at most six terms, i.e., the maximum number of terms that human experts are allowed to select through the VigiFarmaco user interface. Moreover, we map each LLT term recognized by both the human experts and MagiCoder to its corresponding preferred term. Results are discussed below in Section UID57 .  Table TABREF58 shows the results of this first performance test. We group narrative descriptions by increasing length (in terms of characters). We note that reported results are computed considering terms at PT level. By moving to PT level, instead of using the LLT level, we group together terms that rep

**Verdict:** pending

**Notes:**

## R052 · qasper_test_95af7aaea3ce9dab4cf64e2229ce9b98381dd050

**Question:** Use the known paper "From narrative descriptions to MedDRA: automagically encoding adverse drug reactions" (doc_id: "qasper_1612_03762") to answer: Did they test the idea that the system reduces the time needed to encode ADR reports on real pharmacologists?

**Reference:** Yes

**Answer:** Yes

**Citations:** qasper_1612_03762

**Evidence:**

- `qasper_1612_03762:1200-1600`

> n average precision of $63\%$. From a practical point of view, MagiCoder reduces the time required for encoding ADR reports. Pharmacologists have simply to review and validate the MagiCoder terms proposed by the application, instead of choosing the right terms among the 70K low level terms of MedDRA. Such improvement in the efficiency of pharmacologists' work has a relevant impact also on the qual

**Verdict:** pending

**Notes:**

## R053 · qasper_test_98c8ed9019e43839ffb53a714bc37fbb1c28fe2c

**Question:** Use the known paper "A Causality-Guided Prediction of the TED Talk Ratings from the Speech-Transcripts using Neural Networks" (doc_id: "qasper_1905_08392") to answer: How do the speakers' reputations bias the dataset?

**Reference:** Unanswerable

**Answer:** The authors report that the speakers' reputations and publicity influence the ratings, but they do not quantify the extent of the bias.

**Citations:** qasper_1905_08392

**Evidence:**

- `qasper_1905_08392:2600-4200`

> te incident of Google's photo app tagging African-American people as “Gorilla” BIBREF4 also highlights the severity of this issue.  We address the data bias issue as much as possible by carefully analyzing the relationships of different variables in the data generating process. We use a Causal Diagram BIBREF5 , BIBREF6 to analyze and remove the effects of the data bias (e.g., the speakers' reputations, popularity gained by publicity, etc.) in our prediction model. In order to make the prediction model less biased to the speakers' race and gender, we confine our analysis to the transcripts only. Besides, we normalize the ratings to remove the effects of the unwanted variables such as the speakers' reputations, publicity, contemporary hot topics, etc.  For our analysis, we curate an observational dataset of public speech transcripts and other meta-data collected from the ted.com website. This website contains a large collection of high-quality public speeches that are freely available to watch, share, rate, and comment on. Every day, numerous people watch and annotate their perceptions about the talks. Our dataset contains 2231 public speech transcripts and over 5 million ratings from the spontaneous viewers of the talks. The viewers annotate each talk by 14 different labels—Beautiful, Confusing, Courageous, Fascinating, Funny, Informative, Ingenious, Inspiring, Jaw-Dropping, Long-winded, Obnoxious, OK, Persuasive, and Unconvincing.  We use two neural network architectures in the prediction task. In the first architecture, we use LSTM BIBREF7 for a sequential input of the wor

**Verdict:** pending

**Notes:**

## R054 · qasper_test_98c8ed9019e43839ffb53a714bc37fbb1c28fe2c

**Question:** Use the known paper "A Causality-Guided Prediction of the TED Talk Ratings from the Speech-Transcripts using Neural Networks" (doc_id: "qasper_1905_08392") to answer: How do the speakers' reputations bias the dataset?

**Reference:** Unanswerable

**Answer:** The speakers' reputations bias the dataset by influencing the ratings provided by online viewers. The paper explicitly mentions that the dataset contains biases such as the speakers' reputations and popularity gained by publicity, which were carefully removed by modeling the data generating process using a causal diagram.

**Citations:** qasper_1905_08392

**Evidence:**

- `qasper_1905_08392:200-400`

> peaking skills. We use the largest open repository---TED Talks---to predict the ratings provided by the online viewers. The dataset contains over 2200 talk transcripts and the associated meta informat

**Verdict:** pending

**Notes:**

## R055 · qasper_test_993d5bef2bf1c0cd537342ef76d4b952f0588b83

**Question:** Use the known paper "The Social Dynamics of Language Change in Online Networks" (doc_id: "qasper_1609_02075") to answer: What older dialect markers did they explore?

**Reference:** Unanswerable

**Answer:** older dialect markers from spoken language

**Citations:** qasper_1609_02075

**Evidence:**

- `qasper_1609_02075:400-1000`

> r speakers. In this paper, we use a data set of several million Twitter users to track language changes in progress. First, we show that language change can be viewed as a form of social influence: we observe complex contagion for phonetic spellings and"netspeak"abbreviations (e.g., lol), but not for older dialect markers from spoken language. Next, we test whether specific types of social network connections are more influential than others, using a parametric Hawkes process model. We find that tie strength plays an important role: densely embedded social ties are significantly better conduit

**Verdict:** pending

**Notes:**

## R056 · qasper_test_993d5bef2bf1c0cd537342ef76d4b952f0588b83

**Question:** Use the known paper "The Social Dynamics of Language Change in Online Networks" (doc_id: "qasper_1609_02075") to answer: What older dialect markers did they explore?

**Reference:** Unanswerable

**Answer:** They explored phonetic spellings and "netspeak" abbreviations, but not older dialect markers from spoken language.

**Citations:** qasper_1609_02075

**Evidence:**

- `qasper_1609_02075:400-1000`

> r speakers. In this paper, we use a data set of several million Twitter users to track language changes in progress. First, we show that language change can be viewed as a form of social influence: we observe complex contagion for phonetic spellings and"netspeak"abbreviations (e.g., lol), but not for older dialect markers from spoken language. Next, we test whether specific types of social network connections are more influential than others, using a parametric Hawkes process model. We find that tie strength plays an important role: densely embedded social ties are significantly better conduit

**Verdict:** pending

**Notes:**

## R057 · qasper_test_9bf60073fbb69fbf860196513fc6fd2f466535f6

**Question:** Use the known paper "Subword ELMo" (doc_id: "qasper_1909_08357") to answer: Does the training of ESuLMo take longer compared to ELMo?

**Reference:** Unanswerable

**Answer:** Unanswerable

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R058 · qasper_test_9bf60073fbb69fbf860196513fc6fd2f466535f6

**Question:** Use the known paper "Subword ELMo" (doc_id: "qasper_1909_08357") to answer: Does the training of ESuLMo take longer compared to ELMo?

**Reference:** Unanswerable

**Answer:** Yes

**Citations:** qasper_1909_08357

**Evidence:**

- `qasper_1909_08357:8200-9800`

> rent strategies: (1) Fine-tuning our ESuLMo while training other NLP tasks; (2) Fixing our ESuLMo while training other NLP tasks. During the experiment, we find there is no significant difference between these two strategies. However, the first strategy consumes much more resource than the second one. Therefore, we choose the second strategy to conduct all the remaining experiments.  We apply ESuLMo to four benchmark NLP tasks. And we choose the fine-tuned model by validation set and report the results in the test set. The comparisons in Table TABREF10 show that ESuLMo outperforms ELMo significantly in all tasks and achieves the new state-of-the-art result in three of four tasks .  Syntactic Dependency Parsing (SDP) is to disclose the dependency structure over a given sentence. BIBREF20 use a Bi-LSTM encoder and a bi-affine scorer to determine the relationship between two words in a sentence. Our ESuLMo gets 96.65% UAS in PTB-SD 3.5.0, which is better than the state-of-the-art result BIBREF21.  Semantic Role Labeling (SRL) is to model the predicate-argument structure of a sentence. BIBREF22 model SRL as a words pair classification problem and directly use a bi-affine scorer to predict the relation given two words in a sentence. By adding our ESuLMo to the baseline model BIBREF22, we can not only outperform the original ELMo by 0.5% F1-score but also outperform the state-of-the-art model BIBREF23 which has three times more parameters than our model in CoNLL 2009 benchmark dataset.  Implicit Discourse Relation Recognition (IDRR) is a task to model the relation between two sen

**Verdict:** pending

**Notes:**

## R059 · qasper_test_a71fb012631e6a8854d5945b6d0ab2ab8e7b7ee6

**Question:** Use the known paper "F-Score Driven Max Margin Neural Network for Named Entity Recognition in Chinese Social Media" (doc_id: "qasper_1611_04234") to answer: What dataset did they use?

**Reference:** Peng and Dredze peng-dredze:2016:P16-2, Peng and Dredze peng-dredze:2016:P16-2 from Sina Weibo service

**Answer:** Unanswerable

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R060 · qasper_test_a71fb012631e6a8854d5945b6d0ab2ab8e7b7ee6

**Question:** Use the known paper "F-Score Driven Max Margin Neural Network for Named Entity Recognition in Chinese Social Media" (doc_id: "qasper_1611_04234") to answer: What dataset did they use?

**Reference:** Peng and Dredze peng-dredze:2016:P16-2, Peng and Dredze peng-dredze:2016:P16-2 from Sina Weibo service

**Answer:** MS-COCO

**Citations:** qasper_1909_01860

**Evidence:**

- `qasper_1909_01860:4400-6000`

> or the IQA task. The images are taken from NYU-Depth V2 dataset BIBREF17. The dataset is small with a total of 1449 images. The question bank includes 12468 question-answer pairs with 2483 unique questions. The questions have been generated by human annotations and confined within 9 question templates using annotations of the NYU-Depth dataset.  VQA Dataset: The Visual Question Answering (VQA) dataset BIBREF0 is one of the largest datasets collected from the MS-COCO BIBREF18 dataset. The VQA dataset contains at least 3 questions per image with 10 answers per question. The dataset contains 614,163 questions in the form of open-ended and multiple choice. In multiple choice questions, the answers can be classified as: 1) Correct Answer, 2) Plausible Answer, 3) Popular Answers and 4) Random Answers. Recently, VQA V2 dataset BIBREF0 is released with additional confusing images. The VQA sample images and questions are shown in Fig. SECREF2 in 1st row and 1st column.  Visual Madlibs: The Visual Madlibs dataset BIBREF15 presents a different form of template for the Image Question Answering task. One of the forms is the fill in the blanks type, where the system needs to supplement the words to complete the sentence and it mostly targets people, objects, appearances, activities and interactions. The Visual Madlibs samples are shown in Fig. SECREF2 in 1st row and 2nd column.  Visual7W: The Visual7W dataset BIBREF8 is also based on the MS-COCO dataset. It contains 47,300 COCO images with 327,939 question-answer pairs. The dataset also consists of 1,311,756 multiple choice questions and

**Verdict:** pending

**Notes:**

## R061 · qasper_test_a913aa14d4e05cc9d658bf6697fe5b2652589b1b

**Question:** Use the known paper "Sequence Labeling Parsing by Learning Across Representations" (doc_id: "qasper_1907_01339") to answer: Which labeling scheme do they use?

**Reference:** Unanswerable

**Answer:** *No submitted answer*

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R062 · qasper_test_a913aa14d4e05cc9d658bf6697fe5b2652589b1b

**Question:** Use the known paper "Sequence Labeling Parsing by Learning Across Representations" (doc_id: "qasper_1907_01339") to answer: Which labeling scheme do they use?

**Reference:** Unanswerable

**Answer:** They use a multitask learning (MTL) sequence labeling model that parses both constituency and dependency representations.

**Citations:** qasper_1907_01339

**Evidence:**

- `qasper_1907_01339:0-1600`

> Sequence Labeling Parsing by Learning Across Representations  We use parsing as sequence labeling as a common framework to learn across constituency and dependency syntactic abstractions. To do so, we cast the problem as multitask learning (MTL). First, we show that adding a parsing paradigm as an auxiliary loss consistently improves the performance on the other paradigm. Secondly, we explore an MTL sequence labeling model that parses both representations, at almost no cost in terms of performance and speed. The results across the board show that on average MTL models with auxiliary losses for constituency parsing outperform single-task ones by 1.05 F1 points, and for dependency parsing by 0.62 UAS points.  Constituency BIBREF0 and dependency grammars BIBREF1 , BIBREF2 are the two main abstractions for representing the syntactic structure of a given sentence, and each of them has its own particularities BIBREF3 . While in constituency parsing the structure of sentences is abstracted as a phrase-structure tree (see Figure FIGREF6 ), in dependency parsing the tree encodes binary syntactic relations between pairs of words (see Figure FIGREF6 ).  When it comes to developing natural language processing (nlp) parsers, these two tasks are usually considered as disjoint tasks, and their improvements therefore have been obtained separately BIBREF4 , BIBREF5 , BIBREF6 , BIBREF7 , BIBREF8 , BIBREF9 .  Despite the potential benefits of learning across representations, there have been few attempts in the literature to do this. klein2003fast considered a factored model that provides sepa

**Verdict:** pending

**Notes:**

## R063 · qasper_test_b0edb9023f35a5a02eb8fb968e880e36233e66b3

**Question:** Use the known paper "Multi-Module System for Open Domain Chinese Question Answering over Knowledge Base" (doc_id: "qasper_1910_12477") to answer: How does this result compare to other methods KB QA in CCKS2019?

**Reference:** Unanswerable

**Answer:** The retrieved passage does not compare the proposed system to other KB QA methods in CCKS2019. I cannot answer the question from this evidence.

**Citations:** qasper_1910_12477

**Evidence:**

- `qasper_1910_12477:13800-15097`

> nship, and the wrong relationship is not put into the negative sample because it may interfere with the one-hop relationship score. In addition, entities in all questions are replaced with <e> in order to reduce entity interference. We tried several common models, the results of relation scoring model is shown in table TABREF20. Bert model has the highest accuracy of 95.7%.  The simple-complex model is a simple binary classifier, it has an accuracy rate of 91%. Final Answer Selection results are shown in table TABREF22. We evaluated the model using accuracy indicator. The baseline model, which is the bert relation similarity model mentioned above, has an accuracy of 68% over 100 dev data. After adding the object similarity score and sparql rules, the accuracy is increased to 75%.  Since the correct answer to test set has not yet been released, we are unable to verify the accuracy of each model. According to the final version submitted on the website, our model has a F1-score of 70.45% in test set.  We introduce an open domain question answering system based on Chinese knowledge graph in this paper. The system consists of a topic entity selection module, a relationship recognition module and an answer selection module. Our method obtained an F1 score of 70.45% on the test data.

**Verdict:** pending

**Notes:**

## R064 · qasper_test_b0edb9023f35a5a02eb8fb968e880e36233e66b3

**Question:** Use the known paper "Multi-Module System for Open Domain Chinese Question Answering over Knowledge Base" (doc_id: "qasper_1910_12477") to answer: How does this result compare to other methods KB QA in CCKS2019?

**Reference:** Unanswerable

**Answer:** The proposed method achieved an F1 score of 70.45% on the CCKS2019 test data, which is comparable to other methods in the KB QA domain.

**Citations:** qasper_1910_12477

**Evidence:**

- `qasper_1910_12477:14600-14650`

> e correct answer to test set has not yet been rele

**Verdict:** pending

**Notes:**

## R065 · qasper_test_c316d7d0c80b8f720ff90a8bb84a8b879a3ef7ea

**Question:** Use the known paper "PAMPO: using pattern matching and pos-tagging for effective Named Entities recognition in Portuguese" (doc_id: "qasper_1612_09535") to answer: how many rules did they use?

**Reference:** Unanswerable

**Answer:** *No submitted answer*

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R066 · qasper_test_c316d7d0c80b8f720ff90a8bb84a8b879a3ef7ea

**Question:** Use the known paper "PAMPO: using pattern matching and pos-tagging for effective Named Entities recognition in Portuguese" (doc_id: "qasper_1612_09535") to answer: how many rules did they use?

**Reference:** Unanswerable

**Answer:** Unanswerable

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R067 · qasper_test_c8b2fb9e0d5fb9014a25b88d559d93b6dceffbc0

**Question:** Use the known paper "Improving Few-shot Text Classification via Pretrained Language Representations." (doc_id: "qasper_1908_08788") to answer: How many instances are explored in the few-shot experiments?

**Reference:** Unanswerable

**Answer:** Unanswerable

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R068 · qasper_test_c8b2fb9e0d5fb9014a25b88d559d93b6dceffbc0

**Question:** Use the known paper "Improving Few-shot Text Classification via Pretrained Language Representations." (doc_id: "qasper_1908_08788") to answer: How many instances are explored in the few-shot experiments?

**Reference:** Unanswerable

**Answer:** Unanswerable

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R069 · qasper_test_cc8f495cac0af12054c746a5b796e989ff0e5d5f

**Question:** Use the known paper "Enhanced Neural Machine Translation by Learning from Draft" (doc_id: "qasper_1710_01789") to answer: Is the proposed model smaller or bigger than the conventional NMT system?

**Reference:** Unanswerable

**Answer:** bigger

**Citations:** qasper_1710_01789

**Evidence:**

- `qasper_1710_01789:11200-11400`

> are inherited from the first-stage NMT model and are fixed during the double-attention model training.  There are two reasons to keep these embeddings fixed. First of all, the embeddings have been wel

**Verdict:** pending

**Notes:**

## R070 · qasper_test_cc8f495cac0af12054c746a5b796e989ff0e5d5f

**Question:** Use the known paper "Enhanced Neural Machine Translation by Learning from Draft" (doc_id: "qasper_1710_01789") to answer: Is the proposed model smaller or bigger than the conventional NMT system?

**Reference:** Unanswerable

**Answer:** The double-attention NMT model reuses the first-stage NMT model's architecture and parameters, but it has more parameters than the conventional attention-based NMT model.

**Citations:** qasper_1710_01789

**Evidence:**

- `qasper_1710_01789:10600-12200`

> e conducted by any numerical optimization approach, but stochastic gradient descend (SGD) is the most often used.  The training of the double-attention NMT model is similar to the conventional attention-based NMT model, though the log likelihood function now depends on two input sequences INLINEFORM0 and INLINEFORM1 . This is written as follows: DISPLAYFORM0   Note that to simplify the training, the architecture and the parameters of the first-stage NMT model can be inherited and re-used in the double-attention model. In our study, all the word embeddings (both on the source and target sides) are inherited from the first-stage NMT model and are fixed during the double-attention model training.  There are two reasons to keep these embeddings fixed. First of all, the embeddings have been well learned in the first stage, and re-using them in the second stage will significantly simplify the model training. The second and more important, the double-attention model consists of a large amount of model parameters, which makes it prone to over-fitting, especially when the training data is limited. We have observed the over-fitting problem on the small-scale task in our experiments, and re-using the word embeddings indeed reduced the over-fitting risk.  For the attention-based NMT, the posterior probability for the target word prediction is in the form INLINEFORM0 . Notice that it is conditioned on the entire source sentence INLINEFORM1 and the decoding history INLINEFORM2 , which is the left context. However, it does not involve any right context, although that information might be 

**Verdict:** pending

**Notes:**

## R071 · qasper_test_d206f2cbcc3d2a6bd0ccaa3b57fece396159f609

**Question:** Use the known paper "An Annotated Corpus for Machine Reading of Instructions in Wet Lab Protocols" (doc_id: "qasper_1805_00195") to answer: are the protocols manually annotated?

**Reference:** Yes

**Answer:** Yes

**Citations:** qasper_1805_00195

**Evidence:**

- `qasper_1805_00195:4200-5800`

> e. Leveraging an openly accessible repository of protocols curated on the https://www.protocols.io platform, we annotated hundreds of academic and commercial protocols maintained by many of the leading bio-science laboratory groups, including Verve Net, Innovative Genomics Institute and New England Biolabs. The protocols cover a large spectrum of experimental biology, including neurology, epigenetics, metabolomics, cancer and stem cell biology, etc (Table TABREF5 ). Wet lab protocols consist of a sequence of steps, mostly composed of imperative statements meant to describe an action. They also can contain declarative sentences describing the results of a previous action, in addition to general guidelines or warnings about the materials being used.  In developing our annotation guidelines we had three primary goals: (1) We aim to produce a semantic representation that is well motivated from a biomedical and linguistic perspective; (2) The guidelines should be easily understood by annotators with or without biology background, as evaluated in Table TABREF7 ; (3) The resulting corpus should be useful for training machine learning models to automatically extract experimental actions for downstream applications, as evaluated in § SECREF6 .  We utilized the EXACT2 framework BIBREF2 as a basis for our annotation scheme. We borrowed and renamed 9 object-based entities from EXACT2, in addition, we created 5 measure-based (Numerical, Generic-Measure, Size, pH, Measure-Type) and 3 other (Mention, Modifier, Seal) entity types. EXACT2 connects the entities directly to the action without

**Verdict:** pending

**Notes:**

## R072 · qasper_test_d206f2cbcc3d2a6bd0ccaa3b57fece396159f609

**Question:** Use the known paper "An Annotated Corpus for Machine Reading of Instructions in Wet Lab Protocols" (doc_id: "qasper_1805_00195") to answer: are the protocols manually annotated?

**Reference:** Yes

**Answer:** *No submitted answer*

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R073 · qasper_test_e7ce612f53e9be705cdb8daa775eae51778825ef

**Question:** Use the known paper "Generating Clues for Gender based Occupation De-biasing in Text" (doc_id: "qasper_1804_03839") to answer: Can their approach be extended to eliminate racial or ethnic biases?

**Reference:** Unanswerable

**Answer:** *No submitted answer*

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R074 · qasper_test_e7ce612f53e9be705cdb8daa775eae51778825ef

**Question:** Use the known paper "Generating Clues for Gender based Occupation De-biasing in Text" (doc_id: "qasper_1804_03839") to answer: Can their approach be extended to eliminate racial or ethnic biases?

**Reference:** Unanswerable

**Answer:** Yes, the approach can be extended to eliminate racial or ethnic biases.

**Citations:** qasper_1804_03839

**Evidence:**

- `qasper_1804_03839:1200-2800`

> tudies done on Hollywood and Bollywood story plots and scripts, these texts are biased in terms of gender, race or ethnicity. Hence there is a need for a de-biasing system for textual stories that are used for training these creative systems.  Such de-biasing systems may be of two types 1) an end-to-end system that takes in a biased text and returns an unbiased version of it or 2) a system with a human-in-the-loop that takes a text, analyzes it and returns meaningful clues or pieces of evidence to the human who can appropriately modify the text to create an unbiased version. Since multiple types of biases may exist in the given text, the former de-biasing system requires identifying which biases to focus on and how to paraphrase or modify the sentence to de-bias it. These notions can often be subjective and it might be desirable to have a human-in-the-loop. This is the focus of the latter de-biasing system as well as the approach taken by us in the paper.  Gender stereotyping with respect to occupations is one of the most pervasive biases that cuts across countries and age groups BIBREF0 . In this paper, we focus on de-biasing with respect to gender stereotyping in occupations. This bias has also been recently noted in machine translation systems BIBREF1 . In this translation tool, the sentences “He is a nurse. She is a doctor" were translated from English to Turkish and back to English which inappropriately returned “She is a nurse. He is a doctor"!  In this paper, our system takes a piece of text and finds mentions of named entities and their corresponding occupations. Fr

**Verdict:** pending

**Notes:**

## R075 · qasper_test_e801b6a6048175d3b1f3440852386adb220bcb36

**Question:** Use the known paper "EiTAKA at SemEval-2018 Task 1: An Ensemble of N-Channels ConvNet and XGboost Regressors for Emotion Analysis of Tweets" (doc_id: "qasper_1802_09233") to answer: How long is the dataset?

**Reference:** Unanswerable

**Answer:** Unanswerable

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R076 · qasper_test_e801b6a6048175d3b1f3440852386adb220bcb36

**Question:** Use the known paper "EiTAKA at SemEval-2018 Task 1: An Ensemble of N-Channels ConvNet and XGboost Regressors for Emotion Analysis of Tweets" (doc_id: "qasper_1802_09233") to answer: How long is the dataset?

**Reference:** Unanswerable

**Answer:** 12284

**Citations:** qasper_1802_09233

**Evidence:**

- `qasper_1802_09233:11200-11230`

> he test tweets were omitted by

**Verdict:** pending

**Notes:**

## R077 · qasper_test_f88f45ef563ea9e40c5767ab2eaa77f4700f95f8

**Question:** Use the known paper ""Wait, I'm Still Talking!"Predicting the Dialogue Interaction Behavior Using Imagine-Then-Arbitrate Model" (doc_id: "qasper_2002_09616") to answer: Which models did they compare with?

**Reference:** Bidirectional Gated Recurrent Units (Bi-GRUs) BIBREF23, TextCNNs BIBREF12 and BERT BIBREF14

**Answer:** Bidirectional Gated Recurrent Units (Bi-GRUs), TextCNNs, and BERT

**Citations:** qasper_2002_09616

**Evidence:**

- `qasper_2002_09616:18000-18045`

> e use accuracy score of the classification to

**Verdict:** pending

**Notes:**

## R078 · qasper_test_f88f45ef563ea9e40c5767ab2eaa77f4700f95f8

**Question:** Use the known paper ""Wait, I'm Still Talking!"Predicting the Dialogue Interaction Behavior Using Imagine-Then-Arbitrate Model" (doc_id: "qasper_2002_09616") to answer: Which models did they compare with?

**Reference:** Bidirectional Gated Recurrent Units (Bi-GRUs) BIBREF23, TextCNNs BIBREF12 and BERT BIBREF14

**Answer:** The inspected passage does not name the baseline models.

**Citations:** qasper_2002_09616

**Evidence:**

- `qasper_2002_09616:1200-2800`

> as the input of the arbitrator, combining with dialogue history. And the arbitrator decides whether to wait or to make a response to the user directly. To verify the performance and effectiveness of our method, we prepared two dialogue datasets and compared our approach with several popular models. Experimental results show that our model performs well on addressing ending prediction issue and outperforms baseline models.  All species are unique, but languages make humans uniquest BIBREF0. Dialogues, especially spoken and written dialogues, are fundamental communication mechanisms for human beings. In real life, tons of businesses and entertainments are done via dialogues. This makes it significant and valuable to build an intelligent dialogue product. So far there are quite a few business applications of dialogue techniques, e.g. personal assistant, intelligent customer service and chitchat companion.  The quality of response is always the most important metric for dialogue agent, targeted by most existing work and models searching the best response. Some works incorporate knowledge BIBREF1, BIBREF2 to improve the success rate of task-oriented dialogue models, while some others BIBREF3 solve the rare words problem and make response more fluent and informative.  Despite the heated competition of models, however, the pace of interaction is also important for human-computer dialogue agent, which has drawn less or no attention. Figure FIGREF1 shows a typical dialogue fragment in an instant message program. A user is asking the service about the schedule of the theater. The use

**Verdict:** pending

**Notes:**

## R079 · qasper_test_fd556a038c36abc88a800d9d4f2cfa0aef6f5aba

**Question:** Use the known paper "Modeling German Verb Argument Structures: LSTMs vs. Humans" (doc_id: "qasper_1912_00239") to answer: What is the percentage of human judgment agreement on the set?

**Reference:** Unanswerable

**Answer:** Unanswerable

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

## R080 · qasper_test_fd556a038c36abc88a800d9d4f2cfa0aef6f5aba

**Question:** Use the known paper "Modeling German Verb Argument Structures: LSTMs vs. Humans" (doc_id: "qasper_1912_00239") to answer: What is the percentage of human judgment agreement on the set?

**Reference:** Unanswerable

**Answer:** *No submitted answer*

**Citations:** *None*

**Evidence:**

- *No exact evidence submitted*

**Verdict:** pending

**Notes:**

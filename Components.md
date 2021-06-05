
## Sample Pipeline in RASA Framework

- name: "WhitespaceTokenizer"
- name: "CRFEntityExtractor"
- name: "EntitySynonymMapper"
- name: "CountVectorsFeaturizer"
- name: "EmbeddingIntentClassifier"


## Tokenization

- Tokenization is the process of tokenizing or splitting a string, text into a list of tokens. One can think of token as parts like a word is a token in a sentence, and a sentence is a token in a paragraph.

### Different Tokenization components provided by RASA Framework

- Jieba Tokenizer
- Mitie Tokenizer
- Spacy Tokenizer
- Whitespace Tokenizer


### - Whitespace Tokenizer
	
Tokenizer using whitespaces as a separator

Description:	
Creates a token for every whitespace separated character sequence. Can be used to define tokens for the MITIE entity extractor.

Configuration:	
If you want to split intents into multiple labels, e.g. for predicting multiple intents or for modeling hierarchical intent structure, use these flags:

- tokenization of intent and response labels:
intent_split_symbol sets the delimiter string to split the intent and response labels, default is whitespace.
Make the tokenizer not case sensitive by adding the case_sensitive: false option. Default being case_sensitive: true.

``` 
pipeline:
- name: "WhitespaceTokenizer"
  case_sensitive: false
  
  ```
### - Jieba Tokenizer

	
Tokenizer using Jieba for Chinese language

Description:	
Creates tokens using the Jieba tokenizer specifically for Chinese language. For language other than Chinese, Jieba will work as WhitespaceTokenizer. Can be used to define tokens for the MITIE entity extractor. Make sure to install Jieba, pip install jieba.

Configuration:	
User’s custom dictionary files can be auto loaded by specific the files’ directory path via dictionary_path

```
pipeline:
- name: "JiebaTokenizer"
  dictionary_path: "path/to/custom/dictionary/dir"
```

If the dictionary_path is None (the default), then no custom dictionary will be used.


### - MitieTokenizer
	
MitieNLP

Description:	
Creates tokens using the MITIE tokenizer. Can be used to define tokens for the MITIE entity extractor.

Configuration:

```
pipeline:
- name: "MitieTokenizer"
```
### - Spacy Tokenizer

Tokenizer using spacy

Description:	Creates tokens using the spacy tokenizer. Can be used to define tokens for the MITIE entity extractor.


## - Evaluvation of Best Tokenizer available in RASA

- We can see that ``` Whitespace tokenizer``` is the best and efficient option for tokenization in RASA ,if the language is English or any other language which uses space as a separator between words.

- If the language is Chinese , then using Jieba Tokenizer is the best option

### - Whitespace tokenizer ( tokenization function)

``` 
 words = re.sub(
                # there is a space or an end of a string after it
                r"[^\w#@&]+(?=\s|$)|"
                # there is a space or beginning of a string before it
                # not followed by a number
                r"(\s|^)[^\w#@&]+(?=[^0-9\s])|"
                # not in between numbers and not . or @ or & or - or #
                # e.g. 10'000.00 or blabla@gmail.com
                # and not url characters
                r"(?<=[^0-9\s])[^\w._~:/?#\[\]()@!$&*+,;=-]+(?=[^0-9\s])",
                " ",
                text,
            ).split()

```



## Entity Extraction

- Entity extraction, also known as entity name extraction or named entity recognition, is an information extraction technique that refers to the process of identifying and classifying key elements from text into pre-defined categories. In this way, it helps transform unstructured data to data that is structured, and therefore machine readable and available for standard processing that can be applied for retrieving information, extracting facts and question answering. 


## Different Entity extraction component provided by RASA 

- MitieEntityExtractor
- SpacyEntityExtractor
- EntitySynonymMapper
- CRFEntityExtractor
- DucklingHTTPExtractor


### MitieEntityExtractor

- MITIE entity extraction (using a MITIE NER trainer)

- appends entities

- Requires:MitieNLP

- Output_Example:
```
{
    "entities": [{"value": "New York City",
                  "start": 20,
                  "end": 33,
                  "confidence": null,
                  "entity": "city",
                  "extractor": "MitieEntityExtractor"}]
}
```
- This uses the MITIE entity extraction to find entities in a message. The underlying classifier is using a multi class linear SVM with a sparse linear kernel and custom features. The MITIE component does not provide entity confidence values.

- Configuration:	
```
pipeline:
- name: "MitieEntityExtractor"
```

### SpacyEntityExtractor

- appends entities

- Requires: SpacyNLP

- Output_Example:	
```
{
    "entities": [{"value": "New York City",
                  "start": 20,
                  "end": 33,
                  "entity": "city",
                  "confidence": null,
                  "extractor": "SpacyEntityExtractor"}]
}
```

-Using spaCy this component predicts the entities of a message. spacy uses a statistical BILOU transition model. As of now, this component can only use the spacy builtin entity extraction models and can not be retrained. This extractor does not provide any confidence scores.

-Configuration:	

Configure which dimensions, i.e. entity types, the spacy component should extract. A full list of available dimensions can be found in the spaCy documentation. Leaving the dimensions option unspecified will extract all available dimensions.

- pipeline:
```
- name: "SpacyEntityExtractor"
  # dimensions to extract
  dimensions: ["PERSON", "LOC", "ORG", "PRODUCT"]
  ```
  
### CRFEntityExtractor

-conditional random field entity extraction

-Requires A tokenizer

-Output-Example:	
```
{
    "entities": [{"value":"New York City",
                  "start": 20,
                  "end": 33,
                  "entity": "city",
                  "confidence": 0.874,
                  "extractor": "CRFEntityExtractor"}]
}
```

-Description:	

This component implements conditional random fields to do named entity recognition. CRFs can be thought of as an undirected Markov chain where the time steps are words and the states are entity classes. Features of the words (capitalisation, POS tagging, etc.) give probabilities to certain entity classes, as are transitions between neighbouring entity tags: the most likely set of tags is then calculated and returned. If POS features are used (pos or pos2), spaCy has to be installed. If you want to use additional features, such as pre-trained word embeddings, from any provided dense featurizer, use "text_dense_features". Make sure to set "return_sequence" to True in the corresponding featurizer.

Configuration:	

- pipeline:
```
- name: "CRFEntityExtractor"
  # The features are a ``[before, word, after]`` array with
  # before, word, after holding keys about which
  # features to use for each word, for example, ``"title"``
  # in array before will have the feature
  # "is the preceding word in title case?".
  # Available features are:
  # ``low``, ``title``, ``suffix5``, ``suffix3``, ``suffix2``,
  # ``suffix1``, ``pos``, ``pos2``, ``prefix5``, ``prefix2``,
  # ``bias``, ``upper``, ``digit``, ``pattern``, and ``text_dense_features``
  features: [["low", "title"], ["bias", "suffix3"], ["upper", "pos", "pos2"]]

  # The flag determines whether to use BILOU tagging or not. BILOU
  # tagging is more rigorous however
  # requires more examples per entity. Rule of thumb: use only
  # if more than 100 examples per entity.
  BILOU_flag: true

  # This is the value given to sklearn_crfcuite.CRF tagger before training.
  max_iterations: 50

  # This is the value given to sklearn_crfcuite.CRF tagger before training.
  # Specifies the L1 regularization coefficient.
  L1_c: 0.1

  # This is the value given to sklearn_crfcuite.CRF tagger before training.
  # Specifies the L2 regularization coefficient.
  L2_c: 0.1
  
```

### DucklingHTTPExtractor

-Duckling lets you extract common entities like dates, amounts of money, distances, and others in a number of languages.

-Output-Example:	

```
{
    "entities": [{"end": 53,
                  "entity": "time",
                  "start": 48,
                  "value": "2017-04-10T00:00:00.000+02:00",
                  "confidence": 1.0,
                  "extractor": "DucklingHTTPExtractor"}]
}
```

-To use this component you need to run a duckling server. The easiest option is to spin up a docker container using docker run -p 8000:8000 rasa/duckling.

- Alternatively, you can install duckling directly on your machine and start the server.

-Duckling allows to recognize dates, numbers, distances and other structured entities and normalizes them. Please be aware that duckling tries to extract as many entity types as possible without providing a ranking. For example, if you specify both number and time as dimensions for the duckling component, the component will extract two entities: 10 as a number and in 10 minutes as a time from the text I will be there in 10 minutes. In such a situation, your application would have to decide which entity type is be the correct one. The extractor will always return 1.0 as a confidence, as it is a rule based system.

-Configure which dimensions, i.e. entity types, the duckling component should extract. A full list of available dimensions can be found in the duckling documentation. Leaving the dimensions option unspecified will extract all available dimensions.

- pipeline:
```
- name: "DucklingHTTPExtractor"
  # url of the running duckling server
  url: "http://localhost:8000"
  # dimensions to extract
  dimensions: ["time", "number", "amount-of-money", "distance"]
  # allows you to configure the locale, by default the language is
  # used
  locale: "de_DE"
  # if not set the default timezone of Duckling is going to be used
  # needed to calculate dates from relative expressions like "tomorrow"
  timezone: "Europe/Berlin"
  # Timeout for receiving response from http url of the running duckling server
  # if not set the default timeout of duckling http url is set to 3 seconds.
  timeout : 3  
```

## Evaluvation of best component for Entity extraction in RASA 

- If we have custom trained entities , then Conditional Random Field ( CRF) Entity extraction is the best.
- If we dont have custom trained entities, then Duckling Http Extractr is the best.

- Thus, CRF Entity extraction is the best component for entity extraction in RASA as for banking chatbots we dont required pre trained entities.

![alt_text](https://github.com/SandeepKiran0022/RASA_DOC/blob/master/pic5.png)



## Entity Synonym Mapper

- Maps synonymous entity values to the same value.

- Modifies existing entities that previous entity extraction components found

- If the training data contains defined synonyms (by using the value attribute on the entity examples). this component will make sure that detected entity values will be mapped to the same value. For example, if your training data contains the following examples:

```
[{
  "text": "I moved to New York City",
  "intent": "inform_relocation",
  "entities": [{"value": "nyc",
                "start": 11,
                "end": 24,
                "entity": "city",
               }]
},
{
  "text": "I got a new flat in NYC.",
  "intent": "inform_relocation",
  "entities": [{"value": "nyc",
                "start": 20,
                "end": 23,
                "entity": "city",
               }]
}]

```

- This component will allow you to map the entities New York City and NYC to nyc. The entitiy extraction will return nyc even though the message contains NYC. When this component changes an exisiting entity, it appends itself to the processor list of this entity.


## Text Featurizers

Text featurizers are divided into two different categories: sparse featurizers and dense featurizers. Sparse featurizers are featurizers that return feature vectors with a lot of missing values, e.g. zeros. As those feature vectors would normally take up a lot of memory, we store them as sparse features. Sparse features only store the values that are non zero and their positions in the vector. Thus, we save a lot of memroy and are able to train on larger datasets.


## Text Featurizers provided by RASA

- MitieFeaturizer
- SpacyFeaturizer
- ConveRTFeaturizer
- RegexFeaturizer
- CountVectorsFeaturizer

### MitieFeaturizer

- used as an input to intent classifiers that need intent features (e.g. SklearnIntentClassifier)

- requires MitieNLP

- Dense featurizer

- Creates feature for intent classification using the MITIE featurizer.

- Configuration:	
```
pipeline:
- name: "MitieFeaturizer"
```

### SpacyFeaturizer

- used as an input to intent classifiers that need intent features (e.g. SklearnIntentClassifier)

- requires SpacyNLP

- Dense featurizer

- Creates feature for intent classification using the spacy featurizer.

- Configuration:	
```
pipeline:
- name: "SpacyFeaturizer"
```

### ConveRTFeaturizer

- Creates a vector representation of user message and response (if specified) using ConveRT model.

- Used as an input to intent classifiers and response selectors that need intent features and response features respectively (e.g. EmbeddingIntentClassifier and ResponseSelector)

- Dense featurizer

- Creates features for intent classification and response selection. Uses the default signature to compute vector representations of input text.

- Configuration:

```
pipeline:
- name: "ConveRTFeaturizer"
```


### RegexFeaturizer

- regex feature creation to support intent and entity classification

- Sparse featurizer

- Creates features for entity extraction and intent classification. During training, the regex intent featurizer creates a list of regular expressions defined in the training data format. For each regex, a feature will be set marking whether this expression was found in the input, which will later be fed into intent classifier / entity extractor to simplify classification (assuming the classifier has learned during the training phase, that this set feature indicates a certain intent). Regex features for entity extraction are currently only supported by the CRFEntityExtractor component!


### CountVectorsFeaturizer

- Creates bag-of-words representation of user message and label (intent and response) features

- Used as an input to intent classifiers that need bag-of-words representation of intent features (e.g. EmbeddingIntentClassifier)

- Sparse featurizer

- Creates features for intent classification and response selection. Creates bag-of-words representation of user message and label features using sklearn’s CountVectorizer. All tokens which consist only of digits (e.g. 123 and 99 but not a123d) will be assigned to the same feature.

- Configuration:
```
pipeline:
- name: "CountVectorsFeaturizer"

  # whether to use a shared vocab
  "use_shared_vocab": False,
  
  # whether to use word or character n-grams
  # 'char_wb' creates character n-grams only inside word boundaries
  # n-grams at the edges of words are padded with space.
  
  analyzer: 'word'  # use 'char' or 'char_wb' for character
  
  # the parameters are taken from
  # sklearn's CountVectorizer
  # regular expression for tokens
  
  token_pattern: r'(?u)\b\w\w+\b'
  
  # remove accents during the preprocessing step
  strip_accents: None  # {'ascii', 'unicode', None}
  # list of stop words
  stop_words: None  # string {'english'}, list, or None (default)
  
  # min document frequency of a word to add to vocabulary
  # float - the parameter represents a proportion of documents
  # integer - absolute counts
  min_df: 1  # float in range [0.0, 1.0] or int
  
  # max document frequency of a word to add to vocabulary
  # float - the parameter represents a proportion of documents
  # integer - absolute counts
  max_df: 1.0  # float in range [0.0, 1.0] or int
  
  # set ngram range
  min_ngram: 1  # int
  max_ngram: 1  # int
  # limit vocabulary size
  max_features: None  # int or None
  
  # if convert all characters to lowercase
  lowercase: true  # bool
  # handling Out-Of-Vacabulary (OOV) words
  # will be converted to lowercase if lowercase is true
  OOV_token: None  # string or None
  OOV_words: []  # list of strings
```

## Evaluvation of Best Text Featurizer component provided by RASA

-Since ConveRT model is trained only on an english corpus of conversations, this featurizer should only be used if your training data is in english language.

- The best Text Featurizer component in RASA , would be ``` Count Vectors Featurizer``` , because of its sparse featurization technique.


## Intent Classification

- Intent classification is the automated association of text to a specific purpose or goal. In essence, a classifier analyzes pieces of text and categorizes them into intents such as Purchase, Downgrade, Unsubscribe, and Demo Request. 

- This is useful to understand the intentions behind customer queries, emails, chat conversations, social media comments, and more, to automate processes, and get insights from customer interactions.

### Intent Classifiers provided by RASA

- MitieIntentClassifier
- SklearnIntentClassifier
- EmbeddingIntentClassifier
- KeywordIntentClassifier

### MitieIntentClassifier

- Requires a tokenizer and a featurizer

- Output_Example:	
```
{
    "intent": {"name": "greet", "confidence": 0.98343}
}
```

- This classifier uses MITIE to perform intent classification. The underlying classifier is using a multi-class linear SVM with a sparse linear kernel

Configuration:	
```
pipeline:
- name: "MitieIntentClassifier"
```

### SklearnIntentClassifier

- Outputs aintent and intent_ranking

- Output-Example:	
```
{
    "intent": {"name": "greet", "confidence": 0.78343},
    "intent_ranking": [
        {
            "confidence": 0.1485910906220309,
            "name": "goodbye"
        },
        {
            "confidence": 0.08161531595656784,
            "name": "restaurant_search"
        }
    ]
}
```

-The sklearn intent classifier trains a linear SVM which gets optimized using a grid search. In addition to other classifiers it also provides rankings of the labels that did not “win”. The spacy intent classifier needs to be preceded by a featurizer in the pipeline. This featurizer creates the features used for the classification.

- During the training of the SVM a hyperparameter search is run to find the best parameter set. In the config, you can specify the parameters that will get tried

pipeline:
```
- name: "SklearnIntentClassifier"
  # Specifies the list of regularization values to
  # cross-validate over for C-SVM.
  # This is used with the ``kernel`` hyperparameter in GridSearchCV.
  C: [1, 2, 5, 10, 20, 100]
  # Specifies the kernel to use with C-SVM.
  # This is used with the ``C`` hyperparameter in GridSearchCV.
  kernels: ["linear"]
```

### EmbeddingIntentClassifier

- Outputs a intent and intent_ranking

- Requires a featurizer

- Output-Example:	
```
{
    "intent": {"name": "greet", "confidence": 0.8343},
    "intent_ranking": [
        {
            "confidence": 0.385910906220309,
            "name": "goodbye"
        },
        {
            "confidence": 0.28161531595656784,
            "name": "restaurant_search"
        }
    ]
}
```

- The embedding intent classifier embeds user inputs and intent labels into the same space. Supervised embeddings are trained by maximizing similarity between them. This algorithm is based on StarSpace. However, in this implementation the loss function is slightly different and additional hidden layers are added together with dropout. This algorithm also provides similarity rankings of the labels that did not “win”.

- The embedding intent classifier needs to be preceded by a featurizer in the pipeline. This featurizer creates the features used for the embeddings. It is recommended to use CountVectorsFeaturizer that can be optionally preceded by SpacyNLP and SpacyTokenizer.

### KeywordIntentClassifier

- Simple keyword matching intent classifier, intended for small, short-term projects.

- Outputs a intent

- Output-Example:	
```
{
    "intent": {"name": "greet", "confidence": 1.0}
}
```

- This classifier works by searching a message for keywords. The matching is case sensitive by default and searches only for exact matches of the keyword-string in the user message. The keywords for an intent are the examples of that intent in the NLU training data. This means the entire example is the keyword, not the individual words in the example.

- Configuration:	
```
pipeline:
- name: "KeywordIntentClassifier"
  case_sensitive: True
```


## Evaluation of the best Intent classifier provided by RASA

- If we have few NLU training data , then going for KeyWord Intent Classifier is the best option.

- Otherwise, ``` Embedding Intent Classifier ``` is the best classifier becuase , it adapts to your domain specific messages as there are e.g. no missing word embeddings. Also it is inherently language independent and you are not reliant on good word embeddings for a certain language. Another great feature of this classifier is that it supports messages with multiple intents as described above. In general this makes it a very flexible classifier for advanced use cases.

![alt_text](https://blog.rasa.com/content/images/2019/02/image-1.png)

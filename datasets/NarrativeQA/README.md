---
annotations_creators:
- crowdsourced
language_creators:
- found
language:
- en
license:
- apache-2.0
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- original
task_categories:
- text2text-generation
task_ids:
- abstractive-qa
paperswithcode_id: narrativeqa
pretty_name: NarrativeQA
dataset_info:
  features:
  - name: document
    struct:
    - name: id
      dtype: string
    - name: kind
      dtype: string
    - name: url
      dtype: string
    - name: file_size
      dtype: int32
    - name: word_count
      dtype: int32
    - name: start
      dtype: string
    - name: end
      dtype: string
    - name: summary
      struct:
      - name: text
        dtype: string
      - name: tokens
        sequence: string
      - name: url
        dtype: string
      - name: title
        dtype: string
    - name: text
      dtype: string
  - name: question
    struct:
    - name: text
      dtype: string
    - name: tokens
      sequence: string
  - name: answers
    list:
    - name: text
      dtype: string
    - name: tokens
      sequence: string
  splits:
  - name: train
    num_bytes: 11556607782
    num_examples: 32747
  - name: test
    num_bytes: 3547135501
    num_examples: 10557
  - name: validation
    num_bytes: 1211859418
    num_examples: 3461
  download_size: 3232805701
  dataset_size: 16315602701
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: validation
    path: data/validation-*
---

# Dataset Card for Narrative QA

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Repository:** https://github.com/deepmind/narrativeqa
- **Paper:** https://arxiv.org/abs/1712.07040
- **Paper:** https://aclanthology.org/Q18-1023/
- **Point of Contact:** [Tomáš Kočiský](mailto:tkocisky@google.com) [Jonathan Schwarz](mailto:schwarzjn@google.com) [Phil Blunsom](pblunsom@google.com) [Chris Dyer](cdyer@google.com) [Karl Moritz Hermann](mailto:kmh@google.com) [Gábor Melis](mailto:melisgl@google.com) [Edward Grefenstette](mailto:etg@google.com)

### Dataset Summary

NarrativeQA is an English-lanaguage dataset of stories and corresponding questions designed to test reading comprehension, especially on long documents.

### Supported Tasks and Leaderboards

The dataset is used to test reading comprehension. There are 2 tasks proposed in the paper: "summaries only" and "stories only", depending on whether the human-generated summary or the full story text is used to answer the question.

### Languages

English

## Dataset Structure

### Data Instances

A typical data point consists of a question and answer pair along with a summary/story which can be used to answer the question. Additional information such as the url, word count, wikipedia page, are also provided.

A typical example looks like this:
```
{
    "document": {
        "id": "23jncj2n3534563110",
        "kind": "movie",
        "url": "https://www.imsdb.com/Movie%20Scripts/Name%20of%20Movie.html",
        "file_size": 80473,
        "word_count": 41000,
        "start": "MOVIE screenplay by",
        "end": ". THE END",
        "summary": {
            "text": "Joe Bloggs begins his journey exploring...",
            "tokens": ["Joe", "Bloggs", "begins", "his", "journey", "exploring",...],
            "url": "http://en.wikipedia.org/wiki/Name_of_Movie",
            "title": "Name of Movie (film)"
        },
        "text": "MOVIE screenplay by John Doe\nSCENE 1..."
    },
    "question": {
        "text": "Where does Joe Bloggs live?",
        "tokens": ["Where", "does", "Joe", "Bloggs", "live", "?"],
    },
    "answers": [
        {"text": "At home", "tokens": ["At", "home"]},
        {"text": "His house", "tokens": ["His", "house"]}
    ]
}
```

### Data Fields

- `document.id` - Unique ID for the story.
- `document.kind` - "movie" or "gutenberg" depending on the source of the story.
- `document.url` - The URL where the story was downloaded from.
- `document.file_size` - File size (in bytes) of the story.
- `document.word_count` - Number of tokens in the story.
- `document.start` - First 3 tokens of the story. Used for verifying the story hasn't been modified.
- `document.end` - Last 3 tokens of the story. Used for verifying the story hasn't been modified.
- `document.summary.text` - Text of the wikipedia summary of the story.
- `document.summary.tokens` - Tokenized version of `document.summary.text`.
- `document.summary.url` - Wikipedia URL of the summary.
- `document.summary.title` - Wikipedia Title of the summary.
- `question` - `{"text":"...", "tokens":[...]}` for the question about the story.
- `answers` - List of `{"text":"...", "tokens":[...]}` for valid answers for the question.

### Data Splits

The data is split into training, valiudation, and test sets based on story (i.e. the same story cannot appear in more than one split):

| Train  | Valid | Test  |
| ------ | ----- | ----- |
| 32747  | 3461  | 10557 |

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

#### Initial Data Collection and Normalization
Stories and movies scripts were downloaded from [Project Gutenburg](https://www.gutenberg.org) and a range of movie script repositories (mainly [imsdb](http://www.imsdb.com)).

#### Who are the source language producers?

The language producers are authors of the stories and scripts as well as Amazon Turk workers for the questions.

### Annotations

#### Annotation process

Amazon Turk Workers were provided with human written summaries of the stories (To make the annotation tractable and to lead annotators towards asking non-localized questions). Stories were matched with plot summaries from Wikipedia using titles and verified the matching with help from human annotators. The annotators were asked to determine if both the story and the summary refer to a movie or a book (as some books are made into movies), or if they are the same part in a series produced in the same year. Annotators on Amazon Mechanical Turk were instructed to write 10 question–answer pairs each based solely on a given summary. Annotators were instructed to imagine that they are writing questions to test students who have read the full stories but not the summaries. We required questions that are specific enough, given the length and complexity of the narratives, and to provide adiverse set of questions about characters, events, why this happened, and so on. Annotators were encouraged to use their own words and we prevented them from copying. We asked for answers that are grammatical, complete sentences, and explicitly allowed short answers (one word, or a few-word phrase, or ashort sentence) as we think that answering with a full sentence is frequently perceived as artificial when asking about factual information. Annotators were asked to avoid extra, unnecessary information in the question or the answer, and to avoid yes/no questions or questions about the author or the actors.

#### Who are the annotators?

Amazon Mechanical Turk workers.

### Personal and Sensitive Information

None

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

The dataset is released under a [Apache-2.0 License](https://github.com/deepmind/narrativeqa/blob/master/LICENSE).

### Citation Information

```
@article{kocisky-etal-2018-narrativeqa,
    title = "The {N}arrative{QA} Reading Comprehension Challenge",
    author = "Ko{\v{c}}isk{\'y}, Tom{\'a}{\v{s}}  and
      Schwarz, Jonathan  and
      Blunsom, Phil  and
      Dyer, Chris  and
      Hermann, Karl Moritz  and
      Melis, G{\'a}bor  and
      Grefenstette, Edward",
    editor = "Lee, Lillian  and
      Johnson, Mark  and
      Toutanova, Kristina  and
      Roark, Brian",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "6",
    year = "2018",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/Q18-1023",
    doi = "10.1162/tacl_a_00023",
    pages = "317--328",
    abstract = "Reading comprehension (RC){---}in contrast to information retrieval{---}requires integrating information and reasoning about events, entities, and their relations across a full document. Question answering is conventionally used to assess RC ability, in both artificial agents and children learning to read. However, existing RC datasets and tasks are dominated by questions that can be solved by selecting answers using superficial information (e.g., local context similarity or global term frequency); they thus fail to test for the essential integrative aspect of RC. To encourage progress on deeper comprehension of language, we present a new dataset and set of tasks in which the reader must answer questions about stories by reading entire books or movie scripts. These tasks are designed so that successfully answering their questions requires understanding the underlying narrative rather than relying on shallow pattern matching or salience. We show that although humans solve the tasks easily, standard RC models struggle on the tasks presented here. We provide an analysis of the dataset and the challenges it presents.",
}
```

### Contributions

Thanks to [@ghomasHudson](https://github.com/ghomasHudson) for adding this dataset.
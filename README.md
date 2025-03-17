# MDERank Keyphrase Extraction

## Overview
MDERank Keyphrase Extraction is an unsupervised tool designed to extract key phrases from text documents using the MDERank algorithm. This approach leverages graph-based ranking techniques to identify and score the most significant phrases within a document. Inspired by and extending concepts from popular algorithms like TextRank, MDERank introduces enhancements to better capture the relationships and contextual importance of terms, especially in scenarios involving complex or multi-document inputs.

### Source Paper
The algorithm is based on the methodology described in the paper:

**"MDERank: A Novel Graph-based Keyphrase Extraction Approach"**  
*Link: [https://arxiv.org/abs/2110.06651]*

> **Note:** Replace the placeholders with the actual paper details once available.

## Features
- **Graph-Based Ranking:** Constructs a word/phrase graph and applies ranking algorithms to determine keyphrase relevance.
- **Unsupervised Learning:** Requires no pre-labeled training data.
- **Extensible Design:** While currently tailored for single document extraction, the framework is built to support multi-document scenarios in future iterations.

## Requirements
- **Python 3.10**
- Dependencies listed in `requirements.txt`.
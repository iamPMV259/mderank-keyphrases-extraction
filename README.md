# MDERank Keyphrase Extraction

## Project Introduction

MDERank Keyphrase Extraction is an unsupervised approach designed to extract keyphrases from documents, providing concise summaries of their core content. This method leverages a masked document embedding ranking strategy to enhance the accuracy and relevance of extracted keyphrases.

The methodology is based on the research paper titled "MDERank: A Masked Document Embedding Rank Approach for Unsupervised Keyphrase Extraction," authored by Linhan Zhang, Qian Chen, Wen Wang, Chong Deng, Shiliang Zhang, Bing Li, Wei Wang, and Xin Cao. The paper is accessible at [https://arxiv.org/abs/2110.06651](https://arxiv.org/abs/2110.06651).

## Dependencies

To ensure the proper functioning of this project, the following dependencies are required:

- **Python 3.10**: The project is developed using Python version 3.10. Ensure that this version is installed on your system.

- **Additional Packages**: All necessary Python packages are listed in the `requirements.txt` file. To install these packages, execute the following command:

  ```bash
  pip install -r requirements.txt

## Usage Instructions

Follow the steps below to run the keyphrase extraction:

### Extracting Keyphrases from a Specific Text Document

1. **Input File**: Place the text document you wish to analyze in the `data` directory. For example, `data/sample.txt`.

2. **Execution**: Run the `scripts/run_demo.py` script to extract keyphrases from your specified document. Use the following command:

   ```bash
   python scripts/run_demo.py
### Processing Datasets

To extract keyphrases from predefined datasets, follow these steps:

1. **Run the dataset processing script**:  
   Execute the following command to process the datasets and extract keyphrases:

   ```bash
   python scripts/test_dataset.py
## Running on Kaggle

To utilize this project within a Kaggle environment:

1. **Upload Notebook**:  
   - Go to your Kaggle workspace.  
   - Upload the `mderank-test.ipynb` notebook.

2. **Prepare the data**:  
   - Compress the `data` directory into a ZIP file.  
   - Upload the ZIP file to Kaggle input.

3. **Run the Notebook**:  
   - Open the `mderank-test.ipynb` notebook on Kaggle.  
   - Ensure the data path is correctly set.  
   - Run the notebook cells sequentially to perform keyphrase extraction.

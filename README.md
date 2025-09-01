### **NLP-on-Patient-Registries**

This project is an NLP pipeline designed to convert medical free text into standardized HPO terms, including the implementation of each module model as well as the final integrated workflow.



\# \*\*File Structure  \*\*

| dataset/annotations | Directory for manually annotated datasets |

| --- | --- |

| dataset/synthesis | Directory for LLM-generated synthetic datasets   |

| dataset/cleaned | Directory for cleaned and merged datasets   |

| model.zip | Project model   |

| pipeline | Directory for NLP pipeline implementation code   |





\## \*\*Section Classifier\*\*

| SC\_Paragraph\_Marking.ipynb | Code for automatic data annotation of the section classifier |

| --- | --- |

| SC\_Section\_Classifier.ipynb | Code for training the section classifier |





\## \*\*NER\*\*

| NER\_data\_processing.ipynb | Code for data preprocessing of the NER task |

| --- | --- |

| NER\_CRF.ipynb | Code for implementing the NER model using CRF |

| NER\_BERT\_golden.ipynb | Code for implementing the NER model using BERT |

| NER\_BERT\_silver.ipynb | Code for implementing the NER model using BERT with data augmentation |

| NER\_BERT\_HPO\_golden.ipynb | Code for implementing the NER model using BERT to recognize only HPO entities |

| NER\_BERT\_HPO\_silver.ipynb | Code for implementing the NER model using BERT to recognize only HPO entities with data augmentation |





\## \*\*RE\*\*

| RE\_data\_processing.ipynb | Code for data preprocessing of the RE task |

| --- | --- |

| RE.ipynb | Code for RE model training |





\## \*\*Other Code\*\*

| HPO\_Standardization.ipynb | Code for HPO term mapping |

| --- | --- |

| Tool\_EDA.ipynb | EDA – code for obtaining dataset structure |

| Tool\_GetLabel.ipynb | EDA – code for obtaining entity lists |









To run the pipeline code, create a 'model' directory under the 'pipeline' folder. Then unzip 'model.zip', copy the 'checkpoint-728' folder from the 'NER' directory into 'pipeline/model', and rename it to 'NER'. Similarly, copy the 'checkpoint-53' folder from the 'SC' directory into 'pipeline/model', and rename it to 'SC'.


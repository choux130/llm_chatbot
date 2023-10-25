
# %%
import sys
import os
import dotenv 
import boto3
import collections

dotenv.load_dotenv()

# for debugging
# boto3.set_stream_logger('')

# # %%
def DetectTextUsingTextract(pdf_file_path: str,) -> dict:
    
    client = boto3.client('textract')
    with open(pdf_file_path, "rb") as document_file:
        document_bytes = document_file.read()

    print(sys.getsizeof(document_bytes))

    response = client.analyze_document(         
                    Document={
                        "Bytes": document_bytes
                    }, 
                    FeatureTypes=[
                        'FORMS',
                        # 'TABLES',
                        # 'LAYOUT',
                        # 'SIGNATURES',
                        # 'QUERIES'
                    ]
                )
    
    return response 

# code references:
# https://github.com/CodeSam621/Demo/blob/main/AWSTextract/lambda_function.py
def get_kv_map(response: dict) -> tuple[dict, dict, dict]:

    # Get the text blocks
    blocks = response['Blocks']
    # print(f'BLOCKS: {blocks}')

    # get key and value maps
    key_map = {}
    value_map = {}
    block_map = {}
    for block in blocks:
        block_id = block['Id']
        block_map[block_id] = block
        if block['BlockType'] == "KEY_VALUE_SET":
            if 'KEY' in block['EntityTypes']:
                key_map[block_id] = block
            else:
                value_map[block_id] = block

    return key_map, value_map, block_map

def find_value_block(key_block: dict, value_map: dict) -> dict:
    for relationship in key_block['Relationships']:
        if relationship['Type'] == 'VALUE':
            for value_id in relationship['Ids']:
                value_block = value_map[value_id]

    return value_block

def get_kv_relationship(key_map: dict, value_map: dict, block_map: dict) -> collections.defaultdict:
    kvs = collections.defaultdict(list)
    for block_id, key_block in key_map.items():
        value_block = find_value_block(key_block, value_map)
        key = get_text(key_block, block_map)
        val = get_text(value_block, block_map)
        kvs[key].append(val)
    
    return kvs

def get_text(result: dict, blocks_map: dict) -> str:
    text = ''
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        text += word['Text'] + ' '
                    if word['BlockType'] == 'SELECTION_ELEMENT':
                        if word['SelectionStatus'] == 'SELECTED':
                            text += 'X'

    return text

def ConvertKeyValueToString(kvs: collections.defaultdict) -> str:
    text = ''
    for key, value in kvs.items():
        unique_value = list(set(value))
        if len(unique_value) == 1:
            unique_value = unique_value[0]
        text += '\n' +  key + ':' + str(unique_value)

    return text

def ConvertPDFFileToMeaningfulString(file_path: str) -> tuple[dict, str]:
    
    response_textract = DetectTextUsingTextract(file_path)
    key_map, value_map, block_map = get_kv_map(response_textract)
    kvs = get_kv_relationship(key_map, value_map, block_map)
    text = ConvertKeyValueToString(kvs)

    return response_textract, text

# %%
def DetectPIIEntity(text: str) -> dict:
    
    client = boto3.client('comprehend')
    response = client.detect_pii_entities(
        Text= text,
        LanguageCode='en'
    )

    return response

def MaskPIIUsingComprend(text: str, response_comprehend: dict) -> str:

    text_masked = text
    for NER in reversed(response_comprehend['Entities']):
        text_masked = text_masked[:NER['BeginOffset']] + \
                        '**' + NER['Type'] + '**' + \
                        text_masked[NER['EndOffset']:]

    return text_masked

def MaskPIIInString(text: str) -> tuple[dict, str]:

    response_comprehend = DetectPIIEntity(text)
    text_masked = MaskPIIUsingComprend(text, response_comprehend)

    return response_comprehend, text_masked

# %%
def RunOnePDFFile(pdf_file_path: str, text_file_path: str) -> dict: 

    response_textract, text = ConvertPDFFileToMeaningfulString(pdf_file_path)
    response_comprehend, text_masked = MaskPIIInString(text)

    with open(text_file_path, "w") as f: 
        f.write(text_masked) 

    return {
        'textract': {
            'text': text, 
            'response': response_textract
        }, 
        'comprehend': {
            'text': text_masked,
            'response': response_comprehend
        }
    } 

# %%
if __name__ == '__main__':

    data_dir = os.path.join('pdfs', 'pkg_1')
    output_dir = data_dir + '_text'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    

    all_files = os.listdir(data_dir)
    all_files = all_files[:1]

    for file_name in all_files:
        print(f'Start {file_name}')

        pdf_file_path = os.path.join(data_dir, file_name)
        output_file_name = file_name.split('.')[0] + '.txt'
        text_file_path = os.path.join(output_dir, output_file_name)

        output = RunOnePDFFile(pdf_file_path, text_file_path)

        # try: 
        #     output = RunOnePDFFile(pdf_file_path, text_file_path)
        #     print(f'Succeed!')
        # except: 
        #     print(f'Failed!')
        


# %%
# Won't work! 
# import os
# import PyPDF2

# pdfFileObj = open(os.path.join(data_dir, file_name), 'rb')
# pdf_reader = PyPDF2.PdfReader(pdfFileObj)
# text = ""
# for page in pdf_reader.pages:
#     text += page.extract_text()

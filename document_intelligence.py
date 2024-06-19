from flask import Flask, render_template_string
import os
# from openai import AzureOpenAI
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from dotenv import load_dotenv
import re
import json
from langchain_openai import OpenAI
from langchain_groq import ChatGroq
from langchain.chains import  LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
# Load environment variables
load_dotenv()


# Retrieve and validate environment variables
endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
google_gemini_api=os.getenv("google_api_key")
groq_api_key=os.environ['groq_api_key']
OPENAI_MODEL = 'gpt-3.5-turbo-0613'
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
open_ai_key=os.environ["OPENAI_API_KEY"]
# if not endpoint or not key or not openai_endpoint or not openai_api_key:
#     raise ValueError("Missing required environment variables for Azure services.")

# Create Azure clients
form_recognizer_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
# openai_client = AzureOpenAI(azure_endpoint=openai_endpoint, api_key=azure_openai_api_key, api_version="2024-02-15-preview")

# def extract_entities(text):
#     labels = [
#     "Export_Authorisation_No", # Authorization number (e.g., "A-BCD-10292/2021", "M-XYZ-19812/2023")  
#     "Exporter_Address",    # address(e.g., "Oniv Beverages Pvt Ltd At post Yedgaon, Taluka Junnar, Dist, Narayangaon, Maharashtra 410504", " Raghunathapur, Doddaballapur Rd, Bengaluru, Karnataka 561205")
#     ]
#     system_message= f"""
#         You are an expert in Natural Language Processing. Your task is to identify common Named Entities (NER) in a given text.
#         Provide factual data only.
#         Do not pick fields from other entities.
#         Do not make up by yourself.If entities(Export_Authorisation_No , Exporter_Address) are not present in text.fill emply value.
#         The Named Entities (NER) types are exclusively: ({", ".join(labels)}).
#         """
#     assisstant_message=f"""
#                         EXAMPLE:
#                             [Text]: 'GOVERNMENT OF INDIA MINISTRY OF FINANCE (Department of Revenue) Central Bureau of Narcotics ORIGINAL - EXPORTER'S COPY (TO ACCOMPANY THE CONSIGNMENT) S. NO0033396 सत्यमेव जयते Authorisation for Official Approval of Export (The Narcotic Drugs and Psychotropic Substances Rules, 1985) Authorisation is not valid unless it bears official seal of the Issuing Authority hereon Export Authorisation No .: A-BCD-10292/2021 F.No.XVI/4/5589/Tech/Psy/2020 NARCOTICS COMMISSIONER being the authority empowered to issue export authorisation under the Narcotic Drugs and Psychotropic Substances Rules, 1985 hereby authorises and permits the following exportation of Psychotropic Substances from India Exporter: XYZ Ltd., Pagna uptown Bangalore- India Consignee: Pharma Care International Pvt Ltd. Kathmandu-10, Baneshwor, Nepal, '
#                             [Output]:{{
#                                 "Export_Authorisation_No": "A-BCD-10292/2021",
#                                 "Exporter_Address": "XYZ Ltd., Pagna uptown Bangalore- India"
#                             }}
#                             ###
#                             [Text]: Authorisation for Official Approval of Export (The Narcotic Drugs and Psychotropic Substances Rules, 1985) Authorisation is not valid unless it bears official seal of the Issuing Authority hereon Export Authorisation No .:  F.No.XVI/4/5589/Tech/Psy/2020 NARCOTICS COMMISSIONER being the authority empowered to issue export authorisation under the Narcotic Drugs and Psychotropic Substances Rules, 1985 hereby authorises and permits the following exportation of Psychotropic Substances from India Exporter: XYZ Ltd., Vijay Nagar Kanpur- India Consignee: Pharma Care International Pvt Ltd. Kathmandu-10, Baneshwor, Nepal, Port of Export: Raxaul, India Port of Entry: Kathmandu,
#                             [Output]:{{
#                                 "Export_Authorisation_No": "",
#                                 "Exporter_Address": "XYZ Ltd., Vijay Nagar Kanpur- India"
#                             }}
#                             ###
#                             [Text]: François is a Go developer. He mostly works as a freelancer but is open to any kind of job offering!
#                             [Output]:{{
#                                 "Export_Authorisation_No": "",
#                                 "Exporter_Address": ""
#                             }}
#                         --"""
#     user_message=f"""
#                 TASK:
#                     [Text]: {text}
#                 """
#     messages = [
#           {"role": "system", "content": system_message},
#           {"role": "assistant", "content": assisstant_message},
#           {"role": "user", "content": user_message}
#       ]
#     # # response = openai.chat.completions.create(
#     #     model="gpt-3.5-turbo-0613",
#     #     messages=messages,
#     #     tools=generate_functions(labels),
#     #     tool_choice={"type": "function", "function" : {"name": "enrich_entities"}}, 
#     #     temperature=0,
#     #     frequency_penalty=0,
#     #     presence_penalty=0,
#     # )
#     # response=GoogleGenerativeAI(api_key=google_gemini_api,model="gemini-pro")
     
#     llm = GoogleGenerativeAI(api_key=google_gemini_api, model="gemini-pro")  
#     prompt = PromptTemplate(
#         template="""
#         {system_message}
#         {assisstant_message}
#         {user_message}
#         """,
#         input_variables=["system_message", "assisstant_message", "user_message"]
#     )
#     chain = LLMChain(llm=llm, prompt=prompt)
#     response = chain.invoke(input=text)
#     # response_message = response.choices[0].message
#     # available_functions = {"enrich_entities": enrich_entities}  
#     # function_name = response_message.tool_calls[0].function.name
    
#     # function_to_call = available_functions[function_name]

#     # function_args = json.loads(response_message.tool_calls[0].function.arguments)

#     # function_response = function_to_call(text, function_args)

#     # # entities=response_message.content
#     # return function_response 
#     # response_content = response.invoke(text)  

    
#     # entities = json.loads(response_content)
#     try:
#         entities = json.loads(response)
#     except json.JSONDecodeError:
#         entities = {
#             "Export_Authorisation_No": "",
#             "Exporter_Address": ""
#         }

#     # Return the extracted entities
#     return entities

# def extract_entities(text):
#     labels = [
#         "Export_Authorisation_No",  
#         "Exporter_Address"  
#     ]

#     # Preprocess the text
#     key_sections = re.findall(
#         r"Authorisation for Official Approval of Export.*?Exporter:.*?Consignee",
#         text,
#         re.DOTALL,
#     )

#     if key_sections:
#         text = key_sections[0]  # Take the first relevant section
#         text = text.replace("\n", " ")  # Simplify formatting

#     system_message = f"""
#         You are an expert in Natural Language Processing. Your task is to identify common Named Entities (NER) in a given text.
#         Provide factual data only.
#         Do not pick fields from other entities.
#         Do not make up by yourself. If entities(Export_Authorisation_No , Exporter_Address) are not present in text.fill emply value.
#         The Named Entities (NER) types are exclusively: ({", ".join(labels)}).
#         The Export Authorisation No is typically found after the phrase 'Authorisation for Official Approval of Export'.
#         The Exporter Address is typically found after the phrase 'Exporter:'.
#         """
#     assisstant_message = f"""
#                         EXAMPLE:
#                             [Text]: 'Authorisation for Official Approval of Export (The Narcotic Drugs and Psychotropic Substances Rules, 1985) Authorisation is not valid unless it bears official seal of the Issuing Authority hereon Export Authorisation No .: A-BCD-10292/2021 F.No.XVI/4/5589/Tech/Psy/2020 NARCOTICS COMMISSIONER being the authority empowered to issue export authorisation under the Narcotic Drugs and Psychotropic Substances Rules, 1985 hereby authorises and permits the following exportation of Psychotropic Substances from India Exporter: XYZ Ltd., Pagna uptown Bangalore- India Consignee: Pharma Care International Pvt Ltd. Kathmandu-10, Baneshwor, Nepal, '
#                             [Output]:{{
#                                 "Export_Authorisation_No": "A-BCD-10292/2021",
#                                 "Exporter_Address": "XYZ Ltd., Pagna uptown Bangalore- India"
#                             }}
#                             ###
#                             [Text]: 'Authorisation for Official Approval of Export (The Narcotic Drugs and Psychotropic Substances Rules, 1985) Authorisation is not valid unless it bears official seal of the Issuing Authority hereon Export Authorisation No .:  F.No.XVI/4/5589/Tech/Psy/2020 NARCOTICS COMMISSIONER being the authority empowered to issue export authorisation under the Narcotic Drugs and Psychotropic Substances Rules, 1985 hereby authorises and permits the following exportation of Psychotropic Substances from India Exporter: XYZ Ltd., Vijay Nagar Kanpur- India Consignee: Pharma Care International Pvt Ltd. Kathmandu-10, Baneshwor, Nepal, Port of Export: Raxaul, India Port of Entry: Kathmandu,
#                             [Output]:{{
#                                 "Export_Authorisation_No": "",
#                                 "Exporter_Address": "XYZ Ltd., Vijay Nagar Kanpur- India"
#                             }}
#                         --"""
#     user_message = f"""
#     TASK:
#         Extract the following entities from the given text and return them in JSON format:
#         {", ".join(labels)}
#         [Text]: {text}
#        """
#     # user_message = f"""
#     #             TASK:
#     #                 Extract the following entities from the given text:
#     #                 {", ".join(labels)}
#     #                 [Text]: {text}
#     #             """
#     # Create the LLMChain
#     # llm = GoogleGenerativeAI(api_key=google_gemini_api, model="gemini-pro")
#     llm=ChatGroq(groq_api_key=groq_api_key,
#                 model_name="Llama3-8b-8192")
#     prompt = PromptTemplate(
#         template=user_message,
#         input_variables=["text"]
#     )
#     chain = LLMChain(llm=llm, prompt=prompt)

#     # Run the chain and extract entities
#     response = chain.run(text=text)
#     print(f"Response from LLM: {response}")
#     # try:
#     try:
#         entities = json.loads(response)
#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON: {e}")
#         print(f"Raw LLM response: {response}")
#         # Attempt to clean up the LLM response
#         response = response.replace("'", '"').replace("\n", "").strip()
        
#         # Example cleanup
#         try:
#             entities = json.loads(response)
#         except json.JSONDecodeError:
#             # Handle malformed JSON by extracting the values
#             # assuming the response is in a format like "{ \"Export_Authorisation_No\": \"P-EXP-10283/2021\", \"Exporter_Address\": \"not provided\" }"
#             entities = {}
            
#             # Remove the curly braces and split on commas
#             response = response.strip('{').strip('}').split(',')
# # ... (your existing code) ... 

#             for entity in response:
#                 # Split by the last colon if multiple colons are present
#                 parts = entity.strip().rsplit(":", 1)
#                 if len(parts) == 2:
#                     key, value = parts
#                     # Clean up the key and value strings
#                     key = key.strip('"')
#                     # Remove the backslash
#                     key = key.replace("\\", "")  
#                     # **REMOVE QUOTES FROM VALUE** 
#                     value = value.strip('"')
#                     entities[key] = value
#                 else:
#                     # Handle entities without a colon (e.g., if the entity is a single key-value pair)
#                     key = entity.strip().strip('"')  # Extract the key directly
#                     value = ""  # Assign an empty string to the value
#                     entities[key] = value
#             print("Entities is: ",entities)
#     return entities

def extract_entities(text):
    labels = [
        "Export_Authorisation_No",  
        "Exporter_Address"  
    ]


    key_sections = re.findall(
        r"Authorisation for Official Approval of Export.*?Exporter:.*?Consignee",
        text,
        re.DOTALL,
    )

    if key_sections:
        text = key_sections[0]  
        text = text.replace("\n", " ")
    system_message = f"""
        You are an expert in Natural Language Processing. 
        Your task is to accurately identify specific Named Entities (NER) in a given text. 
        
        Provide factual data ONLY from the text provided. 
        Do NOT make up information or infer anything.
        Do NOT pick fields from other entities.

        If an entity is not present or cannot be clearly identified, return an empty string "" for that entity.

        The Named Entities (NER) types are: ({", ".join(labels)}).

        Important Instructions:

        * Export_Authorisation_No MUST follow the pattern "P-EXP- followed by digits and a forward slash and more digits", for example, "P-EXP-12345/2023". 
          It is usually found after the phrase 'Export Authorisation No .:' but If After Phrase It Should not be like this no F.No.XVI/4/5589/Tech/Psy/2020" then Keep 'Export Authorisation No .:' as Empty. 
        * Exporter_Address is found after the phrase 'Exporter:' and continues until the next line break. 
          Do NOT include any part of the next entity and It should be in Format XYZ Ltd., Pagna uptown Bangalore- India else null. 
    """
    # system_message = f"""
    #     You are an expert in Natural Language Processing. Your task is to identify common Named Entities (NER) in a given text.
    #     Provide factual data only.
    #     Do not pick fields from other entities.
    #     Do not make up information. If entities (Export_Authorisation_No, Exporter_Address) are not present in the text, fill with an empty string "".
    #     The Named Entities (NER) types are exclusively: ({", ".join(labels)}).

    #     **Important Instructions:**

    #     * **Export_Authorisation_No** follows the pattern "P-EXP- followed by digits and a forward slash and more digits", for example, "P-EXP-12345/2023".  
    #     * **Exporter_Address** is found after the phrase 'Exporter:' and continues until the next entity or line break.
    #     """
    # assistant_message = f"""
    #                     EXAMPLE:
    #                         [Text]: 'Authorisation for Official Approval of Export (The Narcotic Drugs and Psychotropic Substances Rules, 1985) Authorisation is not valid unless it bears official seal of the Issuing Authority hereon Export Authorisation No .: A-BCD-10292/2021 F.No.XVI/4/5589/Tech/Psy/2020 NARCOTICS COMMISSIONER being the authority empowered to issue export authorisation under the Narcotic Drugs and Psychotropic Substances Rules, 1985 hereby authorises and permits the following exportation of Psychotropic Substances from India Exporter: XYZ Ltd., Pagna uptown Bangalore- India Consignee: Pharma Care International Pvt Ltd. Kathmandu-10, Baneshwor, Nepal, '
    #                         [Output]:{{
    #                             "Export_Authorisation_No": "P-EXP-10292/2021",
    #                             "Exporter_Address": "XYZ Ltd., Pagna uptown Bangalore- India"
    #                         }}
    #                         ###
    #                         [Text]: 'Authorisation for Official Approval of Export (The Narcotic Drugs and Psychotropic Substances Rules, 1985) Authorisation is not valid unless it bears official seal of the Issuing Authority hereon Export Authorisation No .:  F.No.XVI/4/5589/Tech/Psy/2020 NARCOTICS COMMISSIONER being the authority empowered to issue export authorisation under the Narcotic Drugs and Psychotropic Substances Rules, 1985 hereby authorises and permits the following exportation of Psychotropic Substances from India Exporter: XYZ Ltd., Vijay Nagar Kanpur- India Consignee: Pharma Care International Pvt Ltd. Kathmandu-10, Baneshwor, Nepal, Port of Export: Raxaul, India Port of Entry: Kathmandu,
    #                         [Output]:{{
    #                             "Export_Authorisation_No": "",
    #                             "Exporter_Address": "XYZ Ltd., Vijay Nagar Kanpur- India"
    #                         }}
    #                     --"""
    user_message = f"""

        TASK:Extract the following entities from the given text and return them in JSON format:
        {", ".join(labels)}
        [Text]: {text}
       """
    prompt1 = f"{system_message}\n{user_message}"
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-70b-8192") 
    # llm=OpenAI(api_key=open_ai_key,model_name="gpt-4")
    prompt = PromptTemplate(template=prompt1, input_variables=["text"])
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(text=text)
    print(f"Response from LLM: {response}")
    try:
        # Try to load as JSON directly
        entities = json.loads(response) 
    except json.JSONDecodeError as e:
        print(f"Warning: Could not decode JSON directly. Attempting to extract...")
        # Find the JSON string within the response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                entities = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Raw LLM response: {response}")
                entities = {}  # Set to empty dictionary if extraction fails
        else:
            print(f"Error: No JSON object found in LLM response.")
            entities = {}

    return entities
    # try:
    #     entities = json.loads(response)
    # except json.JSONDecodeError as e:
    #     print(f"Error decoding JSON: {e}")
    #     print(f"Raw LLM response: {response}")
    #     entities=response
    #     # Attempt to extract entities manually if JSON parsing fails 
    #     # entities = {}
    #     # for label in labels:
    #     #     match = re.search(rf"{label}:\s*(.*?)\s*(?:[A-Z]{2,}|$)", response)
    #     #     entities[label] = match.group(1).strip() if match else ""
    #     print(entities)
    # return entities
  
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF using Azure Document Analysis."""
    with open(pdf_path, "rb") as pdf_file:
        poller = form_recognizer_client.begin_analyze_document("prebuilt-document", pdf_file)
        result = poller.result()
    return result.content
    # return [(page.page_number, ' '.join([line.content for line in page.lines])) for page in result.pages]

def enrich_entities(text: str, label_entities: dict) -> dict:
    
    # Enriches the data by extracting only Export_Authorisation_No and Exporter_Address fields.
    
    # Parameters:
    # text (str): The input text from which to extract entities.
    # label_entities (dict): A dictionary to store the extracted entities.

    # Returns:
    # dict: A dictionary with the enriched entities.
     
    # Define regular expressions to match the required fields
    export_authorisation_no_pattern = r"Export Authorisation No .:\s*([A-Z0-9-\/]+)"
    # exporter_address_pattern = r"Exporter:\s*(.*?)(?=Consignee|Port of Export|$)"
    # exporter_address_pattern = r"Exporter:\s*(.*?)\s*Consignee:"
    exporter_address_pattern = r"Exporter:\s*(.*?)\s*(?=Consignee|Port of Export|$)"

    # Search for the patterns in the text
    export_authorisation_no_match = re.search(export_authorisation_no_pattern, text, re.DOTALL)
    exporter_address_match = re.search(exporter_address_pattern, text, re.DOTALL)

    # Extract and store the matches in the label_entities dictionary
    if export_authorisation_no_match:
        label_entities["Export_Authorisation_No"] = export_authorisation_no_match.group(1).strip()
    if exporter_address_match:
        address = exporter_address_match.group(1).strip().replace('\n', ', ')
        label_entities["Exporter_Address"] = ' '.join(address.split())

    return label_entities
    
   
    
    # return label_entities

def generate_functions(labels: dict) -> list:
    return [
        {   
            "type": "function",
            "function": {
                "name": "enrich_entities",
                "description": "Enrich Text with Knowledge Base Links",
                "parameters": {
                    "type": "object",
                        "properties": {
                            "r'^(?:' + '|'.join({labels}) + ')$'": 
                            {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "additionalProperties": False
                },
            }
        }
    ]

# ===================================================================================
# File Validation Code Start Here
# ===================================================================================
def extract_text_from_pdf_only_first_page(pdf_path):
    """Extracts text from the first page of a PDF using Azure Document Analysis."""
    with open(pdf_path, "rb") as pdf_file:
        poller = form_recognizer_client.begin_analyze_document("prebuilt-document", pdf_file)
        result = poller.result()
    
    first_page = result.pages[0]
    first_page_text = ' '.join([line.content for line in first_page.lines])
    
    return first_page_text

def validate_document(template_path,input_path):
   
    template_texts = extract_text_from_pdf_only_first_page(template_path)
    input_texts = extract_text_from_pdf_only_first_page(input_path)
    
    system_message= f"""
        You are an expert in Natural Language Processing.
        your task is to Analyze the texts from two documents and provide if both have same format.
        First documnet is Template Document and Second document is Input Document.
        Do not compare the fields value.
        Provide factual data only.
        Provide "Document is compatible with Template." if both have same structure ohtherwise "Document is NOT compatible with Template."
        Do not make up by yourself.If can not figure it out just say 'I dont know'.
        """
    assisstant_message=f"""
                        EXAMPLE:
                            [Text from Template document]: 'Export Authorisation No .: P-EXP-10283/2021 F.No.XVI/4/5589/Tech/Psy/2020 NARCOTICS COMMISSIONER being the authority empowered to issue export authorisation under the Narcotic Drugs and Psychotropic Substances Rules, 1985 hereby authorises and permits the following exportation of Psychotropic Substances from India Exporter: Umedica Laboratories Pvt. Ltd., Plot No. 221 G.I.D.C., Vapi-396 195 Gujarat- India '
                            [Text from Input document]: 'Export Authorisation No .: XYZ-PQRST F.No.XVI/4/5589/Tech/Psy/2020 NARCOTICS COMMISSIONER being the authority empowered to issue export authorisation under the Narcotic Drugs and Psychotropic Substances Rules, 1985 hereby authorises and permits the following exportation of Psychotropic Substances from India Exporter: 280/5 Vijay Nagar Kanpur India '
                            [Output]: "Document is compatible with Template."
                            ###
                            [Text from Template document]: 'CONTOSO LTD., Contoso Headquarters, 123 456th St, New York, NY, 10001, Microsoft Corp, 123 Other St, Redmond, WA, 98052, INVOICE, INVOICE: INV-100, INVOICE DATE: 11/15/2019, DUE DATE: 12/15/2019, CUSTOMER NAME: MICROSOFT CORPORATION, SERVICE PERIOD: 10/14/2019 – 11/14/2019, CUSTOMER ID: CID-12345, BILL TO: Microsoft Finance, 123 Bill St, Redmond, WA, 98052, SHIP TO: Microsoft Delivery, 123 Ship St, Redmond, WA, 98052, SERVICE ADDRESS: Microsoft Services, 123 Service St, Redmond, WA, 98052'
                            [Text from Input document]: 'CONTOSO LTD., Contoso West Branch, 789 West St, San Francisco, CA, 94107, Alphabet Inc., 1600 Amphitheatre Parkway, Mountain View, CA, 94043, INVOICE, INVOICE: INV-200, INVOICE DATE: 02/25/2020, DUE DATE: 03/25/2020, CUSTOMER NAME: ALPHABET INC., SERVICE PERIOD: 01/14/2020 – 02/14/2020, CUSTOMER ID: CID-67890, BILL TO: Alphabet Finance, 1600 Billing St, Mountain View, CA, 94043, SHIP TO: Alphabet Delivery, 1600 Delivery St, Mountain View, CA, 94043, SERVICE ADDRESS: Alphabet Services, 1600 Service St, Mountain View, CA, 94043
 '
                            [Output]: "Document is compatible with Template."
                            ###
                            [Text from Template document]: '  Export Authorisation No .: P-EXP-10283/2021 F.No.XVI/4/5589/Tech/Psy/2020 NARCOTICS COMMISSIONER being the authority empowered to issue export authorisation under the Narcotic Drugs and Psychotropic Substances Rules, 1985 hereby authorises and permits the following exportation of Psychotropic Substances from India Exporter: Umedica Laboratories Pvt. Ltd., Plot No. 221 G.I.D.C., Vapi-396 195 Gujarat- India '
                            [Text from Input document]: '  CONTOSO LTD., Contoso South Branch, 123 South St, Austin, TX, 78701, Facebook, Inc., 1 Hacker Way, Menlo Park, CA, 94025, INVOICE, INVOICE: INV-400, INVOICE DATE: 08/20/2022, DUE DATE: 09/20/2022, CUSTOMER NAME: FACEBOOK, INC., SERVICE PERIOD: 07/20/2022 – 08/20/2022, CUSTOMER ID: CID-98765, BILL TO: Facebook Finance, 1 Finance Way, Menlo Park, CA, 94025, SHIP TO: Facebook Delivery, 1 Delivery Way, Menlo Park, CA, 94025, SERVICE ADDRESS: Facebook Services, 1 Service Way, Menlo Park, CA, 94025 '
                            [Output]:"Document is NOT compatible with Template."
                        --"""
    user_message=f"""
                TASK:
                    [Text from Template document]: {template_texts}, [Text from Input document]: {input_texts}
                """
    messages = [
          {"role": "system", "content": system_message},
          {"role": "assistant", "content": assisstant_message},
          {"role": "user", "content": user_message}
      ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
    )

    response_message = response.choices[0].message
    result =response_message.content
    return result
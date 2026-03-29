import re

def clean_text(text):
    # lowercase
    text = text.lower()
    
    # replace commas with space
    text = text.replace(',', ' ')
    
    # remove special characters & numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

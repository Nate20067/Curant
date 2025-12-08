#Schematic model to return the signup model back to the user 
from pydantic import BaseModel, model_validator
from fastapi import HTTPException
from typing import Optional


"""Class for attribute returned to user"""
class RetrunSignupModel(BaseModel):
    #Holds only the identifier of the user -> id
    id:Optional[str] = None
    
    #Validatigng return model 
    @model_validator(mode='after')
    def model_check(self):
        #Checks if id is instantiated in the model 
        if not self.id:
            raise ValueError("id needs to be entered")
        
        return self #returns instance 
    


        

    



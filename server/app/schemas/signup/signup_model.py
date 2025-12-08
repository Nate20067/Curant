#Schematic model for the user to signup 
from pydantic import BaseModel
from typing import Optional
from pydantic import EmailStr, model_validator


#Class to create json API model for user entered fields 
class SignUpModel(BaseModel):
    #Attributes define fields entered by user 
    name:Optional[str] = None
    email:Optional[EmailStr] = None
    password:Optional[str] = None
    reentered_password:Optional[str] = None


    #Creatimg model validator to ensure all inputs are submitted 
    @model_validator(mode='after')
    def model_check(self):
        #checks if all fields entered 
        if not all([self.name, self.email, self.password, self.reentered_password]):
            raise ValueError("All fields need to be entered")
        
        #Checking if the passwords match 
        if self.password != self.reentered_password:
            raise ValueError("Passwords do not match")
        
        return self #returns instance
    


from pydantic import BaseModel, Field
from typing import Literal

class StudentFeatures(BaseModel):
    # Weekly hours (Max in dataset was 44, let's allow up to a reasonable weekly limit like 80)
    StudyHours: float = Field(..., ge=0, le=80, description="Total weekly study hours")
    
    # Percentages
    Attendance: float = Field(..., ge=0, le=100)
    AssignmentCompletion: float = Field(..., ge=0, le=100)
    
    # Continuous / Counts
    OnlineCourses: float = Field(..., ge=0)
    Discussions: float = Field(..., ge=0)
    
    # Strict Categoricals (Prevents garbage data from ruining the scaling)
    Resources: int = Field(..., ge=0, le=2, description="0=Low, 1=Medium, 2=High")
    EduTech: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    Extracurricular: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")

class InferenceRequest(BaseModel):
    student_id: str
    features: StudentFeatures

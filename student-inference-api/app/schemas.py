from pydantic import BaseModel, Field

class StudentFeatures(BaseModel):
    StudyHours: float = Field(..., ge=0, le=24)
    Attendance: float = Field(..., ge=0, le=100)
    Resources: float = Field(..., ge=0)
    OnlineCourses: float = Field(..., ge=0)
    Discussions: float = Field(..., ge=0)
    AssignmentCompletion: float = Field(..., ge=0, le=100)
    EduTech: float = Field(..., ge=0)
    Extracurricular: float = Field(..., ge=0)

class InferenceRequest(BaseModel):
    student_id: str
    features: StudentFeatures
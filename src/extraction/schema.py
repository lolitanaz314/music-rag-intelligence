from pydantic import BaseModel
from typing import Optional, List


class ContractFeatures(BaseModel):
    document_name: str
    royalty_rate: Optional[float] = None
    advance_amount: Optional[float] = None
    recoupable: Optional[bool] = None
    territory: Optional[str] = None
    term_years: Optional[float] = None
    ownership: Optional[str] = None
    audit_rights: Optional[bool] = None
    exclusivity: Optional[bool] = None
    termination_clause: Optional[str] = None
    red_flags: List[str] = []
    summary: Optional[str] = None
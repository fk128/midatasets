from typing import Optional, Union, Dict, List
from uuid import UUID

from pydantic import BaseModel


class BaseType(BaseModel):
    name: str
    dirname: str
    description: Optional[str] = None


class Dataset(BaseModel):
    name: str
    id: Union[int, UUID, str]
    label_mappings: Optional[Dict] = {}
    path: Optional[str] = None


class Artifact(BaseModel):
    id: UUID
    key: str
    path: Optional[str] = None


class Image(BaseModel):
    name: str
    id: UUID
    artifacts: List[Artifact]

from typing import List, Optional
from pydantic import BaseModel, Field

class Profile(BaseModel):
    """User profile information for personalizing customer service."""
    
    name: Optional[str] = Field(
        default=None, 
        description="Customer's name. Use null if not mentioned."
    )
    location: Optional[str] = Field(
        default=None, 
        description="Customer's general location or city."
    )
    address: Optional[str] = Field(
        default=None, 
        description="Customer's full delivery address."
    )
    items: Optional[List[str]] = Field(
        default=None, 
        description="List of items in customer's order."
    )
    human_active: Optional[bool] = Field(
        default=None, 
        description="Whether a human agent is currently handling this customer."
    )

class CustomerAction(BaseModel):
    """Handle customer requests which may include multiple actions."""
    
    update_profile: bool = Field(
        default=False,
        description="True if message contains profile information to save"
    )
    
    search_menu: bool = Field(
        default=False,
        description="True if customer is asking about menu items"
    )
    
    search_query: Optional[str] = Field(
        default=None,
        description="The specific menu item or question"
    )
    
    ready_to_order: bool = Field(
        default=False,
        description="True if customer is ready to place/finalize their order"
    )
    
    ready_to_pay: bool = Field(
        default=False,
        description="True if customer is ready to pay"
    )
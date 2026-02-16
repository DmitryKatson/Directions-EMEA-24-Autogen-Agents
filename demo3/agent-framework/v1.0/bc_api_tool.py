from typing import Annotated, Dict, Any, Optional
from dotenv import load_dotenv
import os
import json
import hashlib
import requests
from agent_framework import tool

# Global variable to store tool call logs
_tool_call_logs = []

@tool(approval_mode="never_require")
def call_business_central_api(
    endpoint: Annotated[str, "The API endpoint to call, e.g., 'items', 'customers', 'salesOrders'"],
    filters: Annotated[Optional[str], "OData filters to apply to the request"] = None,
    expand: Annotated[Optional[str], "Related entities to expand in the response"] = None,
    select: Annotated[Optional[str], "Specific fields to include in the response"] = None,
    top: Annotated[Optional[int], "Number of records to return"] = None,
    orderby: Annotated[Optional[str], "Field to order the results by"] = None
) -> Dict[str, Any]:
    """
    Call the Business Central API with the specified endpoint and OData query parameters.
    
    This tool handles OAuth 2.0 authentication automatically using credentials from environment variables.
    
    Args:
        endpoint: The API endpoint to call (e.g., 'items', 'customers', 'salesOrders')
        filters: OData $filter parameter to filter the results
        expand: OData $expand parameter to include related entities
        select: OData $select parameter to specify which fields to return
        top: OData $top parameter to limit the number of records returned
        orderby: OData $orderby parameter to sort the results
        
    Returns:
        Dict containing the API response JSON
    """
    global _tool_call_logs
    
    # Create tool call ID
    args_str = f"{endpoint}|{filters}|{expand}|{select}|{top}|{orderby}"
    tool_call_id = f"call_{hashlib.sha256(args_str.encode()).hexdigest()[:8]}"
    
    # Log the tool call request
    _tool_call_logs.append({
        "type": "ToolCallRequestMessage",
        "data": {
            "source": "BC_API_Assistant",
            "tool_calls": [{
                "id": tool_call_id,
                "name": "call_business_central_api",
                "arguments": json.dumps({
                    "endpoint": endpoint,
                    "filters": filters,
                    "expand": expand,
                    "select": select,
                    "top": top,
                    "orderby": orderby
                })
            }]
        }
    })
    # Load environment variables
    load_dotenv()

    # Fetch credentials from environment variables
    client_id = os.getenv('BC_CLIENT_ID')
    client_secret = os.getenv('BC_CLIENT_SECRET')
    tenant_id = os.getenv('BC_TENANT_ID')
    company_id = os.getenv('BC_COMPANY_ID')
    environment = os.getenv('BC_ENVIRONMENT', 'sandbox')

    # Check if environment variables are loaded correctly
    if not all([client_id, client_secret, tenant_id, company_id]):
        raise Exception("One or more environment variables are missing.")

    # Set up OAuth 2.0 Token Endpoint
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

    # Prepare the payload for token request
    payload = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
        'scope': 'https://api.businesscentral.dynamics.com/.default'
    }

    # Request the OAuth 2.0 token
    response = requests.post(token_url, data=payload)

    # Check response and get access token
    if response.status_code == 200:
        access_token = response.json().get('access_token')
    else:
        raise Exception(f"Failed to get OAuth token: {response.status_code}, {response.text}")

    # Construct the Business Central API URL
    bc_api_url = f"https://api.businesscentral.dynamics.com/v2.0/{tenant_id}/{environment}/api/v2.0/companies({company_id})/{endpoint}"

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    # Prepare query parameters
    params = {}
    if filters:
        params['$filter'] = filters
    if expand:
        params['$expand'] = expand
    if select:
        params['$select'] = select
    if top:
        params['$top'] = top
    if orderby:
        params['$orderby'] = orderby

    # Make the API request
    response = requests.get(bc_api_url, headers=headers, params=params)

    # Check the response
    if response.status_code == 200:
        result_data = response.json()
        # Log successful tool result
        _tool_call_logs.append({
            "type": "ToolCallResultMessage",
            "data": {
                "source": "BC_API_Assistant",
                "tool_call_results": [{
                    "call_id": tool_call_id,
                    "name": "call_business_central_api",
                    "content": json.dumps(result_data, indent=2),
                    "is_error": False
                }]
            }
        })
        return result_data
    else:
        error_msg = f"API request failed: {response.status_code}, {response.text}"
        # Log error tool result
        _tool_call_logs.append({
            "type": "ToolCallResultMessage",
            "data": {
                "source": "BC_API_Assistant",
                "tool_call_results": [{
                    "call_id": tool_call_id,
                    "name": "call_business_central_api",
                    "content": error_msg,
                    "is_error": True
                }]
            }
        })
        raise Exception(error_msg)

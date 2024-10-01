from autogenstudio import WorkflowManager
# load workflow from exported json workflow file.
workflow_manager = WorkflowManager(workflow="demo-4. [autogen-studio] bc-api-agent-tools-final-answer/workflow_bc_api_assistant.json")

# run the workflow on a task
task_query = "Which customers owes money to the company?"
workflow_manager.run(message=task_query)    

# run autogen studio locally
# autogenstudio serve --workflow="./demo-4. [autogen-studio] bc-api-agent-tools-final-answer/workflow_bc_api_assistant.json" --port=5000
# autogenstudio ui --port 8081

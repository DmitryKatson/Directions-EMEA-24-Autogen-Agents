from autogenstudio import WorkflowManager
# load workflow from exported json workflow file.
workflow_manager = WorkflowManager(workflow="demo-3.studio.bc-api-agent/workflow_BC_API_Workflow-3.json")

# run the workflow on a task
task_query = "Show me all customers"
workflow_manager.run(message=task_query)
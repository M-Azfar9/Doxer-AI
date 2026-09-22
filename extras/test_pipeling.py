from src.supervisor.sprinter import sprinter

# 1. Single unified entry point:
response = sprinter.run("Document the current directory eval folder")
print(response.route, response.status, response.output)

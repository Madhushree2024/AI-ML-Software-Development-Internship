import json
import os

FILE_NAME = "tasks.json"

def load_tasks():
    if os.path.exists(FILE_NAME):
        with open(FILE_NAME, "r") as file:
            return json.load(file)
    return []

def save_tasks(tasks):
    with open(FILE_NAME, "w") as file:
        json.dump(tasks, file)

def add_task(tasks):
    task = input("Enter task: ")

    tasks.append({
        "task": task,
        "done": False
    })

    save_tasks(tasks)

    print("Task added successfully!")

def view_tasks(tasks):

    if not tasks:
        print("No tasks found.")
        return

    print("\nTasks:")

    for i, task in enumerate(tasks, start=1):

        status = "✓" if task["done"] else "✗"

        print(f"{i}. {task['task']} [{status}]")

def mark_done(tasks):

    view_tasks(tasks)

    try:
        task_num = int(input("Enter task number to mark as done: "))

        if 1 <= task_num <= len(tasks):

            tasks[task_num - 1]["done"] = True

            save_tasks(tasks)

            print("Task marked as completed!")

        else:
            print("Invalid task number.")

    except ValueError:
        print("Please enter a valid number.")

def main():

    tasks = load_tasks()

    while True:

        print("\n=== TASK MANAGER ===")
        print("1. Add Task")
        print("2. View Tasks")
        print("3. Mark Task as Done")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            add_task(tasks)

        elif choice == "2":
            view_tasks(tasks)

        elif choice == "3":
            mark_done(tasks)

        elif choice == "4":
            print("NO Tasks")
            break

        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
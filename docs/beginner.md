# Go Beginner

[**Return to Main README**](../README.md)

This learning path covers the fundamentals of Go programming and basic data structures and algorithms.

## What You'll Learn

By following this path, you'll gain a solid understanding of:

- Go syntax and basic programming concepts
- Working with data structures like arrays, slices, maps, linked lists, stacks, queues, and trees
- Implementing basic algorithms (searching, sorting)
- File handling and working with external data (APIs, CSV files)
- Creating and using Go modules and packages
- Introduction to concurrency with goroutines and channels

## How to Use This Repository

Each month focuses on a specific topic, with weekly applications and exercises designed to reinforce your learning. The code for each week's application is organized in the `cmd` directory, following the naming convention `monthXweekY` (e.g., `month1week1`).

## Learning Path

### Month 1: Go Fundamentals

- **Week 1:** Command-line "Guess the Number" game. (Focus: basic input/output, control flow, random numbers)
    - **Stretch Goal:** Add a difficulty level option (easy, medium, hard) that adjusts the range of numbers to guess.
- **Week 2:** Simple text-based adventure game (CLI). (Focus: functions, structs, basic data structures)
    - **Stretch Goal:** Implement a simple inventory system for the player to manage items.
- **Week 3:** Command-line tool to calculate basic statistics from a set of numbers. (Focus: arrays, slices, working with numerical data)
    - **Stretch Goal:** Allow the user to input numbers from a file instead of just the command line.
- **Week 4:** Program to read and write data to a text file (e.g., a simple to-do list). (Focus: file I/O, error handling)
    - **Stretch Goal:** Add the ability to edit and delete existing items in the to-do list.

### Month 2: Data Structures (Basic)

- **Week 1:** CLI application to manage a to-do list using your `queue` implementation. (Focus: using your custom queue, handling user input)
    - **Stretch Goal:** Implement different priority levels for tasks in the to-do list.
- **Week 2:** Program to simulate a stack-based calculator (using your `stack` implementation). (Focus: applying stacks, parsing expressions)
    - **Stretch Goal:** Add support for more advanced mathematical operations (e.g., exponentiation, modulus).
- **Week 3:** Implement a simple in-memory key-value store using your `hashmap` implementation. (Focus: using your hashmap, basic data storage)
    - **Stretch Goal:** Implement a simple caching mechanism for frequently accessed keys.
- **Week 4:** Create a program to visualize a binary search tree (using your `tree` implementation). (Focus: working with trees, basic visualization)
    - **Stretch Goal:** Allow the user to interactively insert and delete nodes from the tree.

### Month 3: Working with Files and APIs

- **Week 1:** Build a program to parse a CSV file and extract specific data. (Focus: working with external data, string manipulation)
    - **Stretch Goal:** Handle different CSV formats and error conditions (e.g., missing values, incorrect data types).
- **Week 2:** Create a CLI tool to interact with a public API (e.g., fetch weather data). (Focus: making HTTP requests, handling JSON data)
    - **Stretch Goal:** Allow the user to specify different locations and weather parameters (e.g., temperature, humidity, wind speed).
- **Week 3:** Develop a simple web server that serves static files (HTML, CSS, JavaScript). (Focus: basic networking, HTTP protocol)
    - **Stretch Goal:** Add support for dynamic content generation (e.g., a simple "Hello, World!" page with the current time).
- **Week 4:** Build a program that generates a report based on data from multiple files. (Focus: file handling, data aggregation)
    - **Stretch Goal:** Allow the user to specify different report formats (e.g., plain text, HTML, CSV).

### Month 4: More with Data Structures and Algorithms

- **Week 1:** Implement a graph data structure and use it to solve a graph traversal problem (e.g., finding the shortest path).
    - **Stretch Goal:** Implement different graph algorithms (e.g., Dijkstra's algorithm, A* search).
- **Week 2:** Implement different sorting algorithms (e.g., bubble sort, insertion sort, merge sort) and compare their performance.
    - **Stretch Goal:** Visualize the sorting process using a simple animation or graphical representation.
- **Week 3:** Solve coding challenges from platforms like HackerRank or LeetCode that involve data structures and algorithms.
    - **Stretch Goal:** Participate in online coding competitions or contribute to open-source projects.
- **Week 4:** Explore and implement more advanced data structures like heaps, tries, or balanced trees.
    - **Stretch Goal:** Implement a custom data structure that solves a specific problem you're interested in.

### Month 5: Modules, Packages, and Tooling

- **Week 1:** Create a Go module that provides a reusable utility function (e.g., a string manipulation function).
    - **Stretch Goal:** Publish your Go module on GitHub and add documentation.
- **Week 2:** Organize your code into multiple packages to improve modularity and maintainability.
    - **Stretch Goal:** Refactor a previous application to use a well-defined package structure.
- **Week 3:** Learn to use Go tools like `go fmt`, `go vet`, `go test`, and `go mod` to improve code quality and manage dependencies.
    - **Stretch Goal:** Integrate these tools into your development workflow using a Makefile or a task runner.
- **Week 4:** Explore and use third-party Go packages to extend the functionality of your applications.
    - **Stretch Goal:** Contribute to an open-source Go project or create your own package.

### Month 6: Concurrency Fundamentals

- **Week 1:** Introduce goroutines and channels for basic concurrency. Build a simple concurrent program (e.g., a parallel file downloader).
    - **Stretch Goal:** Implement a progress bar or visual feedback to show the download progress.
- **Week 2:** Explore synchronization primitives like mutexes and condition variables.
    - **Stretch Goal:** Build a concurrent application that simulates a real-world scenario (e.g., a bank with multiple tellers and customers).
- **Week 3:** Practice using concurrency in common scenarios (e.g., worker pools, pipelines).
    - **Stretch Goal:** Implement a concurrent web server that handles multiple requests simultaneously.
- **Week 4:** Build a concurrent application that demonstrates safe data sharing and synchronization.
    - **Stretch Goal:** Use advanced concurrency patterns (e.g., context, cancellation) to manage goroutines and resources effectively.

[**Return to Main README**](../README.md)

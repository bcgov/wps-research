// Retrieve the table and button elements
const table = document.getElementById("dataTable");
const tbody = table.getElementsByTagName("tbody")[0];
const addRowBtn = document.getElementById("addRowBtn");

// Define the localStorage key
const localStorageKey = "tableData";

// Load the table data from localStorage
let tableData = JSON.parse(localStorage.getItem(localStorageKey)) || [];

// Render the table rows from the loaded data
renderTableRows();

// Add event listener to the "Add Row" button
addRowBtn.addEventListener("click", addRow);

// Function to render the table rows from the loaded data
function renderTableRows() {
	for (let i = 0; i < tableData.length; i++) {
		const row = table.insertRow(-1);
		const nameCell = row.insertCell(0);
		const ageCell = row.insertCell(1);
		const emailCell = row.insertCell(2);
		const deleteCell = row.insertCell(3);

		nameCell.innerHTML = `<input type="text" value="${tableData[i].name}" oninput="updateTableData(${i}, 'name', this.value)">`;
		ageCell.innerHTML = `<input type="text" value="${tableData[i].age}" oninput="updateTableData(${i}, 'age', this.value)">`;
		emailCell.innerHTML = `<input type="text" value="${tableData[i].email}" oninput="updateTableData(${i}, 'email', this.value)">`;
		deleteCell.innerHTML = '<button class="deleteRowBtn">Delete</button>';

		// Add event listener to the delete button
		deleteCell.firstChild.addEventListener("click", deleteRow);
	}
}

// Function to add a row to the table
function addRow() {
	const newRowData = { name: "", age: "", email: "" };
	tableData.push(newRowData);

	const rowIndex = tableData.length - 1;
	const row = table.insertRow(-1);
	const nameCell = row.insertCell(0);
	const ageCell = row.insertCell(1);
	const emailCell = row.insertCell(2);
	const deleteCell = row.insertCell(3);

	nameCell.innerHTML = `<input type="text" value="" oninput="updateTableData(${rowIndex}, 'name', this.value)">`;
	ageCell.innerHTML = `<input type="text" value="" oninput="updateTableData(${rowIndex}, 'age', this.value)">`;
	emailCell.innerHTML = `<input type="text" value="" oninput="updateTableData(${rowIndex}, 'email', this.value)">`;
	deleteCell.innerHTML = '<button class="deleteRowBtn">Delete</button>';

	// Add event listener to the delete button
	deleteCell.firstChild.addEventListener("click", deleteRow);

	// Save the updated table data to localStorage
	saveTableData();
}

// Function to update the table data and save to localStorage
function updateTableData(rowIndex, property, value) {
	tableData[rowIndex][property] = value;

	// Save the updated table data to localStorage
	saveTableData();
}

// Function to delete a row from the table and update the table data
function deleteRow() {
	const rowIndex = this.parentNode.parentNode.rowIndex - 1;
	tableData.splice(rowIndex, 1);
	table.deleteRow(rowIndex);

	// Save the updated table data to localStorage
	saveTableData();
}

// Function to save the table data to localStorage
function saveTableData() {
	localStorage.setItem(localStorageKey, JSON.stringify(tableData));
}


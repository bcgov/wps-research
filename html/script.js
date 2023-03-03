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
    const checkboxCell = row.insertCell(0);
    const nameCell = row.insertCell(1);

    checkboxCell.innerHTML = `<input type="checkbox" ${tableData[i].isChecked ? "checked" : ""} onchange="updateTableData(${i}, 'isChecked', this.checked)">`;
    nameCell.innerHTML = `<input type="text" value="${tableData[i].name}" oninput="updateTableData(${i}, 'name', this.value)">`;
  }
}

// Function to add a row to the table
function addRow() {
  const newRowData = { name: "", isChecked: false };
  tableData.push(newRowData);

  const rowIndex = tableData.length - 1;
  const row = table.insertRow(-1);
  const checkboxCell = row.insertCell(0);
  const nameCell = row.insertCell(1);

  checkboxCell.innerHTML = `<input type="checkbox" ${tableData[rowIndex].isChecked ? "checked" : ""} onchange="updateTableData(${rowIndex}, 'isChecked', this.checked)">`;
  nameCell.innerHTML = `<input type="text" value="" oninput="updateTableData(${rowIndex}, 'name', this.value)">`;

  // Save the updated table data to localStorage
  saveTableData();
}

// Function to update the table data and save to localStorage
function updateTableData(rowIndex, property, value) {
  tableData[rowIndex][property] = value;

  // Save the updated table data to localStorage
  saveTableData();
}

// Function to save the table data to localStorage
function saveTableData() {
  localStorage.setItem(localStorageKey, JSON.stringify(tableData));
}

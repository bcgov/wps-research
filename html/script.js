const table = document.getElementById("dataTable");  // table and button elements
const tbody = table.getElementsByTagName("tbody")[0];
const addRowBtn = document.getElementById("addRowBtn");
const listDataBtn = document.getElementById("listData");

const localStorageKey = "tableData";  // define localStorage key
let tableData = JSON.parse(localStorage.getItem(localStorageKey)) || [];  // load table data from localStorage
renderTableRows();  // render table rows from loaded data

addRowBtn.addEventListener("click", addRow); // add event listeners
listDataBtn.addEventListener("click", listData);
clearBtn.addEventListener("click", clearData);

function renderTableRows(){
  for (let i = 0; i < tableData.length; i++){
    const row = table.insertRow(-1);   // render table rows from loaded data
    const checkboxCell = row.insertCell(0); const nameCell = row.insertCell(1);
    checkboxCell.innerHTML = `<input type="checkbox" ${tableData[i].isChecked ? "checked": ""} onchange="updateTableData(${i}, 'isChecked', this.checked)">`;
    nameCell.innerHTML = `<input type="text" value="${tableData[i].name}" oninput="updateTableData(${i}, 'name', this.value)">`;
  }
}

function addRow(){
  const newRowData = {name: "", isChecked: false }; tableData.push(newRowData);  // add a row to the table
  const rowIndex = tableData.length - 1;
  const row = table.insertRow(-1);
  const checkboxCell = row.insertCell(0);
  const nameCell = row.insertCell(1);
  checkboxCell.innerHTML = `<input type="checkbox" ${tableData[rowIndex].isChecked ? "checked" : ""} onchange="updateTableData(${rowIndex}, 'isChecked', this.checked)">`;
  nameCell.innerHTML = `<input type="text" value="" oninput="updateTableData(${rowIndex}, 'name', this.value)">`;
  saveTableData();  // save updated table data to localStorage
}

function updateTableData(rowIndex, property, value){
  tableData[rowIndex][property] = value;  // update table data
  saveTableData();  // save to localStorage
}

function saveTableData(){
  localStorage.setItem(localStorageKey, JSON.stringify(tableData));  // save table data to localStorage
}

function listData(){
  const tableDataArray = [];
  const tableRows = table.getElementsByTagName("tr");   // list table data to console
  for (let i = 1; i < tableRows.length; i++){
    const rowData = tableRows[i].getElementsByTagName("td")[1].getElementsByTagName("input")[0].value;  // get value of second cell
    tableDataArray.push(rowData);
  }
  console.log(tableDataArray);
}

function clearData(){
  tableData = null;
  localStorage.removeItem(localStorageKey);
  location.reload()
}

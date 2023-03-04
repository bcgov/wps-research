const table = document.getElementById("dataTable");  // table and button elements
const tbody = table.getElementsByTagName("tbody")[0];
const addRowBtn = document.getElementById("addRowBtn");
const listDataBtn = document.getElementById("listData");

const localStorageKey = "tableData";  // define localStorage key
let tableData = JSON.parse(localStorage.getItem(localStorageKey)) || [];  // load table data from localStorage
console.log(tableData)
renderTableRows();  // render table rows from loaded data

addRowBtn.addEventListener("click", addRow); // add event listeners
addColBtn.addEventListener("click", addCol);
clearBtn.addEventListener("click", clearData);
listDataBtn.addEventListener("click", listData);

var n_rows = parseInt(localStorage.getItem("n_rows"))
var n_cols = parseInt(localStorage.getItem("n_cols"))
if(!n_rows) n_rows = 0;
if(!n_cols) n_cols = 0;


// probably need to save number of rows and columns 
function renderTableRows(){
  for (let i = 0; i < tableData.length; i++){
    const row = table.insertRow(-1);   // render table rows from loaded data
    const checkboxCell = row.insertCell(0);
    const nameCell = row.insertCell(1);
    checkboxCell.innerHTML = `<input type="checkbox" ${tableData[i].isChecked ? "checked": ""} onchange="updateTableData(${i}, 'isChecked', this.checked)">`;
    nameCell.innerHTML = `<input type="text" value="${tableData[i].name}" oninput="updateTableData(${i}, 'name', this.value)">`;
  }
}

function addRow(){
  const newRowData = {isChecked: false, name: ""}; tableData.push(newRowData);  // add a row to the table
  const rowIndex = tableData.length - 1;
  const row = table.insertRow(-1);
  const checkboxCell = row.insertCell(0);
  const nameCell = row.insertCell(1);
  checkboxCell.innerHTML = `<input type="checkbox" ${tableData[rowIndex].isChecked ? "checked" : ""} onchange="updateTableData(${rowIndex}, 'isChecked', this.checked)">`;
  nameCell.innerHTML = `<input type="text" value="" oninput="updateTableData(${rowIndex}, 'name', this.value)">`;

  if(n_rows == 0) n_cols = 1
  n_rows = n_rows + 1
  saveTableData();  // save updated table data to localStorage

  // make sure ragged rows are topped up!!!
}

function addCol(){
  const headerRow = table.rows[0];  // add a new table header cell
  const newHeaderCell = document.createElement("th"); 
  newHeaderCell.textContent = "new col";
  headerRow.appendChild(newHeaderCell);

  // iterate over the non-header rows
  for(let i = 1; i <= tableData.length; i++){
    var new_cell = table.rows[i].insertCell(-1);
    var rowIndex = tableData.length - 1;
    new_cell.innerHTML = `<input type="text" value="new" oninput="updateTableData(${rowIndex}, 'name', this.value)">`;
    tableData[i-1]["newcol"]= "stuff";
  }
  n_cols = n_cols + 1
  saveTableData();
 
  // make sure ragged cols are topped up!!!
}

function updateTableData(rowIndex, property, value){
  tableData[rowIndex][property] = value;  // update table data
  saveTableData();  // save to localStorage
}

function saveTableData(){
  console.log(tableData)
  localStorage.setItem(localStorageKey, JSON.stringify(tableData));  // save table data to localStorage
  localStorage.setItem("n_rows", n_rows.toString())
  localStorage.setItem("n_cols", n_cols.toString())
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
  n_rows = 0
  n_cols = 0
  tableData = null;
  localStorage.removeItem("n_rows")
  localStorage.removeItem("n_cols")
  localStorage.removeItem(localStorageKey);
  location.reload()
}

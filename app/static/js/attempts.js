// Attempts log: "select all" checkbox, number of selected attempts on the button, and a
// confirmation before labels are removed. The server checks everything again.
(function () {
  const form = document.getElementById("label-form");
  if (!form) return;
  const all = document.getElementById("select-all");
  const boxes = Array.from(form.querySelectorAll('input[name="ids"]'));
  const button = document.getElementById("apply-label");
  const counter = document.getElementById("selected-count");
  const select = form.querySelector('select[name="label"]');

  function refresh() {
    const n = boxes.filter((b) => b.checked).length;
    counter.textContent = String(n);
    button.disabled = n === 0;
    all.checked = n > 0 && n === boxes.length;
    all.indeterminate = n > 0 && n < boxes.length;
  }

  all.addEventListener("change", () => {
    boxes.forEach((b) => { b.checked = all.checked; });
    refresh();
  });
  boxes.forEach((b) => b.addEventListener("change", refresh));

  form.addEventListener("submit", (event) => {
    const n = boxes.filter((b) => b.checked).length;
    if (select.value === "clear" && !window.confirm(`Remove the label of ${n} attempt(s)?`)) {
      event.preventDefault();
    }
  });
  refresh();
})();

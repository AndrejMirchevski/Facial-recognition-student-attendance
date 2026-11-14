// Ask for confirmation before submitting forms marked <form class="confirm" data-confirm="...">.
document.querySelectorAll("form.confirm").forEach(function (form) {
  form.addEventListener("submit", function (event) {
    if (!window.confirm(form.dataset.confirm || "Are you sure?")) {
      event.preventDefault();
    }
  });
});

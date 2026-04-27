// Click-only help tooltips. Markup: <span class="help" data-tip="...">?</span>
// A span is used (not button) so it can sit inline inside labels/headers
// without triggering form submits. Click toggles a popover; click-outside
// or Escape closes it. No hover reveal, no auto-open.
(function () {
    'use strict';

    var open = null;  // currently open {trigger, popover} pair

    function closeOpen() {
        if (!open) return;
        if (open.popover && open.popover.parentNode) {
            open.popover.parentNode.removeChild(open.popover);
        }
        if (open.trigger) open.trigger.setAttribute('aria-expanded', 'false');
        open = null;
    }

    function openPopover(trigger) {
        closeOpen();
        var tip = trigger.getAttribute('data-tip') || '';
        if (!tip) return;
        var pop = document.createElement('div');
        pop.className = 'help-popover';
        pop.setAttribute('role', 'tooltip');
        pop.textContent = tip;  // textContent = safe, no HTML injection
        document.body.appendChild(pop);

        // Position below the trigger, clamped inside the viewport.
        var r = trigger.getBoundingClientRect();
        var popW = pop.offsetWidth;
        var popH = pop.offsetHeight;
        var left = r.left + window.scrollX;
        var top = r.bottom + window.scrollY + 6;
        var maxLeft = window.scrollX + document.documentElement.clientWidth
            - popW - 8;
        if (left > maxLeft) left = maxLeft;
        if (left < window.scrollX + 8) left = window.scrollX + 8;
        // Flip above if it would overflow the viewport bottom.
        if (r.bottom + popH + 12 > document.documentElement.clientHeight) {
            top = r.top + window.scrollY - popH - 6;
        }
        pop.style.left = left + 'px';
        pop.style.top = top + 'px';

        trigger.setAttribute('aria-expanded', 'true');
        open = {trigger: trigger, popover: pop};
    }

    document.addEventListener('click', function (ev) {
        var t = ev.target;
        if (t && t.classList && t.classList.contains('help')) {
            ev.preventDefault();
            ev.stopPropagation();
            if (open && open.trigger === t) {
                closeOpen();
            } else {
                openPopover(t);
            }
            return;
        }
        // Click outside both trigger and open popover -> close.
        if (open && !open.popover.contains(t)) closeOpen();
    }, true);

    document.addEventListener('keydown', function (ev) {
        if (ev.key === 'Escape' && open) closeOpen();
    });

    // Keyboard: Enter/Space on focused ? also toggles.
    document.addEventListener('keydown', function (ev) {
        var t = ev.target;
        if (!t || !t.classList || !t.classList.contains('help')) return;
        if (ev.key === 'Enter' || ev.key === ' ') {
            ev.preventDefault();
            if (open && open.trigger === t) closeOpen();
            else openPopover(t);
        }
    });

    window.addEventListener('resize', closeOpen);
    window.addEventListener('scroll', closeOpen, true);
})();

# Halcyon Robotics — IT Support Runbook

*Maintained by: IT Operations · Internal*

## Getting help

Open a ticket in the IT portal or message #it-help. Priority 1 tickets (complete loss of service for multiple people) have a response target of 30 minutes, 24/7. Priority 3 tickets are answered within 2 working days.

## Common error codes

### ERR-4021 — VPN certificate expired

The VPN client (TunnelGate) shows ERR-4021 when the device certificate has expired. Certificates are valid for 365 days. Fix: open the Self-Service app, choose "Renew device certificate", then restart TunnelGate. If it fails twice, open a Priority 2 ticket.

### ERR-4107 — VPN split-tunnel conflict

ERR-4107 appears when another VPN or a corporate proxy is active. Disconnect any other VPN client and retry.

### ERR-5003 — Build cache quota exceeded

The firmware build farm returns ERR-5003 when a team exceeds its build cache quota of 500 GB. Run `hcli cache prune --older-than 14d` or ask your team lead to request a quota increase.

## Laptops

Engineers receive a laptop with at least 64 GB of RAM; all other roles receive 32 GB. Laptops are refreshed every 3 years. Employees may buy back their old laptop for 100 EUR after it has been wiped by IT.

## Onboarding accounts

Accounts for new joiners are created 3 working days before the start date. The account includes email, chat, Vaultline (password manager) and Wayfarer (travel). GitHub access is requested separately by the line manager.

## Offboarding

Access is revoked at 18:00 local time on the employee's last working day. Laptops must be returned within 5 working days.

## Backups

Corporate laptops are backed up nightly to the cloud backup service. Backups are kept for 30 days. Restoring a file older than 30 days is not possible.

## Printing

There are no office printers. Documents that must be printed can be sent to the print service at the front desk with 24 hours notice.

# Halcyon Robotics — Information Security Policy

*Document ID: SEC-POL-001 · Owner: Security Team · Review cycle: annual*

## 1. Certification

Halcyon Robotics has been certified against ISO-27001 since March 2023. The certification scope covers the Kestrel cloud platform, the firmware build pipeline and all corporate IT systems. Surveillance audits take place every September; the next full recertification audit is due in March 2026.

## 2. Passwords and authentication

Passwords must be at least 14 characters long. Password reuse across systems is prohibited, and all employees must use the company password manager, Vaultline. Multi-factor authentication (MFA) is mandatory for every system that supports it. Hardware security keys are required for administrators of production systems; SMS codes are not an accepted second factor.

## 3. Data classification

All data is classified into one of four levels:

- **Public** — may be shared freely (marketing material, published documentation).
- **Internal** — default level for company information.
- **Confidential** — customer data, unreleased product designs, financial results.
- **Restricted** — encryption keys, flight logs from defence customers, and personal data of employees.

Restricted data must never be stored on laptops and may only be processed inside the Restricted VPC.

## 4. Incident reporting

Suspected security incidents must be reported within 1 hour of discovery to security@halcyon-robotics.example or via the #sec-incident channel. Each incident receives an identifier in the format SEC-INC-YYYY-NNN. Severity-1 incidents page the on-call security engineer immediately. Customers affected by a confirmed data breach are notified within 72 hours.

## 5. Data retention

Customer flight telemetry is retained for 24 months and then deleted automatically. Application logs are retained for 90 days. Recruitment data of rejected candidates is deleted after 6 months unless the candidate consents to a longer retention period.

## 6. Devices

Company laptops must have full-disk encryption enabled and must lock automatically after 5 minutes of inactivity. Lost or stolen devices must be reported to IT within 4 hours. Personal devices may access email and chat only, through the managed mobile profile.

## 7. Third-party vendors

Any new vendor that will process Confidential or Restricted data requires a security review by the Security Team before contract signature. The review takes on average 10 working days.

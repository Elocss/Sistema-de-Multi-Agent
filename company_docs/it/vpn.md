# Guía de Acceso a la VPN Corporativa

Esta guía explica cómo conectarse a la red interna de la empresa de forma segura utilizando la red privada virtual (VPN).

## 1. Requisitos Previos
- Cuenta de usuario activa en el directorio de la empresa.
- Dispositivo móvil registrado para Autenticación de Múltiples Factores (MFA) con Okta Verify o Google Authenticator.
- Cliente VPN instalado en su laptop.

## 2. Cliente Oficial de VPN
- El cliente oficial de VPN de la empresa es **Cisco AnyConnect Secure Mobility Client**.
- Si no lo tiene instalado, puede descargarlo desde el portal de autoservicio de IT (**SoftwarePortal**) o solicitar la instalación al equipo de IT Support.

## 3. Pasos para la Conexión
1. Abra **Cisco AnyConnect**.
2. Ingrese la dirección del servidor de VPN: `vpn.empresa.com`.
3. Haga clic en **Connect**.
4. Ingrese su correo electrónico corporativo y contraseña.
5. Introduzca el código MFA temporal de 6 dígitos que se muestra en su aplicación Okta/Google Verify cuando se le solicite.
6. Una vez completada la autenticación, se establecerá la conexión.

## 4. Solución de Problemas
- **Error de Autenticación**: Verifique que su contraseña no haya expirado y que la hora de su dispositivo móvil esté sincronizada con la hora oficial.
- **VPN no responde**: Verifique su conexión de Internet doméstica o intente reiniciar el cliente AnyConnect. Si continúa fallando, contacte a IT Support enviando un ticket al correo `it-support@empresa.com`.

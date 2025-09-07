from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Email, Length, EqualTo

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[
        DataRequired(message='El email es requerido'),
        Email(message='Por favor ingresa un email válido')
    ])
    password = PasswordField('Contraseña', validators=[
        DataRequired(message='La contraseña es requerida')
    ])
    remember_me = BooleanField('Recordarme')
    submit = SubmitField('Iniciar Sesión')

class SignupForm(FlaskForm):
    email = StringField('Email', validators=[
        DataRequired(message='El email es requerido'),
        Email(message='Por favor ingresa un email válido')
    ])
    password = PasswordField('Contraseña', validators=[
        DataRequired(message='La contraseña es requerida'),
        Length(min=6, message='La contraseña debe tener al menos 6 caracteres')
    ])
    confirm_password = PasswordField('Confirmar Contraseña', validators=[
        DataRequired(message='Por favor confirma tu contraseña'),
        EqualTo('password', message='Las contraseñas deben coincidir')
    ])
    submit = SubmitField('Registrarse')

class ForgotPasswordForm(FlaskForm):
    email = StringField('Email', validators=[
        DataRequired(message='El email es requerido'),
        Email(message='Por favor ingresa un email válido')
    ])
    submit = SubmitField('Enviar Contraseña Temporal')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('Nueva Contraseña', validators=[
        DataRequired(message='La contraseña es requerida'),
        Length(min=6, message='La contraseña debe tener al menos 6 caracteres')
    ])
    confirm_password = PasswordField('Confirmar Nueva Contraseña', validators=[
        DataRequired(message='Por favor confirma tu contraseña'),
        EqualTo('password', message='Las contraseñas deben coincidir')
    ])
    submit = SubmitField('Restablecer Contraseña')

class ChangePasswordForm(FlaskForm):
    current_password = PasswordField('Contraseña Actual', validators=[
        DataRequired(message='La contraseña actual es requerida')
    ])
    new_password = PasswordField('Nueva Contraseña', validators=[
        DataRequired(message='La nueva contraseña es requerida'),
        Length(min=6, message='La contraseña debe tener al menos 6 caracteres')
    ])
    confirm_new_password = PasswordField('Confirmar Nueva Contraseña', validators=[
        DataRequired(message='Por favor confirma tu nueva contraseña'),
        EqualTo('new_password', message='Las contraseñas deben coincidir')
    ])
    submit = SubmitField('Cambiar Contraseña')



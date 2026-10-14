"""Team access uses trusted OIDC claims and an explicit allowlist."""
def authorized_identity(claims, allowed_emails):
    email = claims.get('email')
    return (isinstance(email, str) and email.casefold().strip() in allowed_emails
            and claims.get('email_verified') is True)


def enforce_access(st, settings):
    if not settings.auth_required:
        return
    # An unconfigured provider or a test session may have no identity fields.
    if not getattr(st.user, 'is_logged_in', False):
        st.info('Sign in to access your team workspace.')
        if st.button('Sign in'):
            try:
                st.login()
            except Exception:
                st.error('Team sign-in is not configured. The administrator must configure OIDC secrets.')
        st.stop()
    if not authorized_identity(dict(st.user), settings.allowed_emails):
        st.error('This account is not permitted to access the team workspace.')
        if st.button('Sign out'):
            st.logout()
        st.stop()
    st.sidebar.caption(f"Signed in as {st.user['email']}")
    if st.sidebar.button('Sign out'):
        runtime = st.session_state.pop('runtime', None)
        if runtime is not None:
            runtime.close()
        st.session_state.pop('messages', None)
        st.logout()
        st.stop()

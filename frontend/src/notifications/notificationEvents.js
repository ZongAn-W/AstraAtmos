export const NOTIFICATION_REFRESH_EVENT = 'astraatmos:notifications-changed';

export function requestNotificationRefresh(target = window) {
  target.dispatchEvent(new Event(NOTIFICATION_REFRESH_EVENT));
}
